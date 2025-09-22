import torch
from torch.autograd import Function
import einops
import triton
import triton.language as tl

import math


@triton.jit
def flash_fwd_kernel(
  Q_ptr, K_ptr, V_ptr,
  O_ptr, L_ptr,
  stride_qb, stride_qq, stride_qd,
  stride_kb, stride_kk, stride_kd,
  stride_vb, stride_vk, stride_vd,
  stride_ob, stride_oq, stride_od,
  stride_lb, stride_lq,
  N_QUERIES, N_KEYS,
  scale,
  D: tl.constexpr,
  Q_TILE_SIZE: tl.constexpr,
  K_TILE_SIZE: tl.constexpr):
  
  # Program indices
  query_tile_index = tl.program_id(0)
  batch_index = tl.program_id(1)

  # Offset each pointer with the corresponding batch index
  # multiplied with the batch stride for each tensor
  Q_block_ptr = tl.make_block_ptr(
    Q_ptr + batch_index * stride_qb,
    shape=(N_QUERIES, D),
    strides=(stride_qq, stride_qd),
    offsets=(query_tile_index * Q_TILE_SIZE, 0),
    block_shape=(Q_TILE_SIZE, D),
    order=(1, 0)
  )
  K_block_ptr = tl.make_block_ptr(
    K_ptr + batch_index * stride_kb,
    shape=(N_KEYS, D),
    strides=(stride_kk, stride_kd),
    offsets=(query_tile_index * K_TILE_SIZE, 0),
    block_shape=(K_TILE_SIZE, D),
    order=(1, 0)
  )
  V_block_ptr = tl.make_block_ptr(
    V_ptr + batch_index * stride_vb,
    shape=(N_KEYS, D),
    stride=(stride_vk, stride_vd),
    offset=(query_tile_index * K_TILE_SIZE, 0),
    block_shape=(K_TILE_SIZE, D),
    order=(1, 0)
  )
  O_block_ptr = tl.make_block_ptr(
    O_ptr + batch_index * stride_ob,
    shape=(N_QUERIES, D),
    stride=(stride_oq, stride_od),
    offset=(query_tile_index * Q_TILE_SIZE, 0),
    block_shape=(Q_TILE_SIZE, D),
    order=(1, 0)
  )
  L_block_ptr = tl.make_block_ptr(
    L_ptr + batch_index * stride_lb,
    shape = (N_QUERIES,),
    stride=(stride_lq,),
    offset=(query_tile_index * Q_TILE_SIZE,),
    block_shape=(Q_TILE_SIZE,),
    order=(0,)
  )

  O = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
  L = tl.zeros((Q_TILE_SIZE, 1), dtype=tl.float32)
  M = tl.full((Q_TILE_SIZE, 1), float("-inf"), dtype=tl.float32)

  Q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
  for _ in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
    K = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
    V = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
    S = tl.dot(Q, K.trans()) * scale  # Q_TILE_SIZE * K_TILE_SIZE
    Mi = tl.maximum(M, tl.max(S, axis=1, keep_dims=True), axis=1)
    m_scale = tl.exp(M - Mi)
    M = Mi
    P = tl.exp(S - M)
    L = m_scale * L + tl.sum(P, axis=1, keep_dims=True)
    O = m_scale * O + tl.dot(Mi, V)

    K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
    V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
  
  tl.store(O_block_ptr, O, boundary_check=(0, 1))
  tl.store(L_block_ptr, (M + tl.log(L)).view((Q_TILE_SIZE,)), boundary_check=(0,))


class FlashAttention2InTriton(Function):
  TILE_SHAPE = (16, 16)

  @staticmethod
  def forward(ctx: Function, Q, K, V, is_casual=False):
    Q_TILE_SIZE, K_TILE_SIZE = FlashAttention2InTriton.TILE_SHAPE
    device = Q.device
    B, Q, D = Q.shape()
    K = K.shape[-2]
    outputs = torch.empty((B, Q, D), device=device)
    l = torch.empty((B, Q), device=device)
    scale =  1 / math.sqrt(D)
    flash_fwd_kernel[(Q // Q_TILE_SIZE, B)](
      Q, K, V,
      outputs, l,
      Q.stride(0), Q.stride(1), Q.stride(2),
      K.stride(0), K.stride(1), K.stride(2),
      V.stride(0), V.stride(1), V.stride(2),
      outputs.stride(0), outputs.stride(1), outputs.stride(2),
      l.stride(0), l.stride(1),
      Q, K,
      scale,
      D,
      Q_TILE_SIZE, K_TILE_SIZE
    )
    ctx.save_for_backward(Q, K, V, l, outputs)
    ctx.is_casual = is_casual
    return outputs

  @staticmethod
  def _backward(ctx, dO):
    Q, K, V, L, O = ctx.saved_tensors
    # bq, bk = tile_shape
    TQ, d_model = Q.shape[-2:]
    TK = K.shape[1]
    d_model_sqrt = math.sqrt(d_model)

    D = einops.reduce(O * dO, "... bq d -> ... bq 1", "sum")
    S = einops.einsum(Q, K, "... bq d, ... bk d -> ... bq bk") / d_model_sqrt
    if ctx.is_casual:
      mask = mask = torch.tril(torch.ones(TQ, TK))
      S = S.masked_fill(mask == 0, float("-inf"))
    P = torch.exp(S - einops.rearrange(L, "... -> ... 1"))
    dV = einops.einsum(P, dO, "... bq bk, ... bq d -> ... bk d")
    dP = einops.einsum(dO, V, "... bq d, ... bk d -> ... bq bk")
    dS = P * (dP - D)
    dQ = einops.einsum(dS, K, "... bq bk, ... bk d -> ... bq d") / d_model_sqrt
    dK = einops.einsum(dS, Q, "... bq bk, ... bq d -> ... bk d") / d_model_sqrt

    return dQ, dK, dV, None
  
  @staticmethod
  def backward(ctx, dO):
    return FlashAttention2InTriton._backward(ctx, dO)