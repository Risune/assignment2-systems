import torch
from torch.autograd import Function
from cs336_basics import utils
import einops
import math


class FlashAttention2InPytorch(Function):
  TILE_SHAPE = (16, 16)

  @staticmethod
  def forward(ctx: Function, Q, K, V, is_casual=False):
    bq, bk = FlashAttention2InPytorch.TILE_SHAPE
    shape = Q.shape[:-2]
    TQ, d_model = Q.shape[-2:]
    TK = K.shape[-2]
    it_tq, it_tk = TQ // bq, TK // bk
    d_model_sqrt = math.sqrt(d_model)

    mask = torch.tril(torch.ones(TQ, TK))

    O = torch.empty(Q.shape)
    L = torch.empty(Q.shape[:-1])

    for i in range(0, it_tq):
      s = i*bq
      Qi = Q[..., s:s+bq, :]
      Oi = torch.zeros((*shape, bq, d_model))
      li = torch.zeros((*shape, bq, 1))
      mi = torch.full((*shape, bq, 1), float("-inf"))
      for j in range(0, it_tk):
        col_s = j*bk
        Kj, Vj = K[..., col_s:col_s+bk, :], V[..., col_s:col_s+bk, :]
        Sij = einops.einsum(Qi, Kj, "... bq d, ... bk d -> ... bq bk") / d_model_sqrt
        if is_casual and s < col_s+bk:
          # mask Sij
          Sij = Sij.masked_fill(mask[s:s+bq, col_s:col_s+bk] == 0, float("-inf"))
        tmp_mi = torch.max(mi, einops.reduce(Sij, "... bq bk -> ... bq 1", "max"))
        scale = torch.exp(mi - tmp_mi)
        mi = tmp_mi
        Pij = torch.exp(Sij - tmp_mi)
        li = scale * li + einops.reduce(Pij, "... bq bk -> ... bq 1", "sum")
        Oi = scale * Oi + einops.einsum(Pij, Vj, "... bq bk, ... bk d -> ... bq d")
      O[..., s:s+bq, :] = Oi / li
      L[..., s:s+bq] = (mi + torch.log(li)).squeeze(-1)
    ctx.save_for_backward(Q, K, V, L, O)
    ctx.is_casual = is_casual
    return O
  
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
  def _tiled_backward(ctx, dO):
    Q, K, V, L, O = ctx.saved_tensors
    bq, bk = FlashAttention2InPytorch.TILE_SHAPE
    shape = Q.shape[:-2]
    TQ, d_model = Q.shape[-2:]
    TK = K.shape[1]
    it_tq, it_tk = TQ // bq, TK // bk
    d_model_sqrt = math.sqrt(d_model)
    D = einops.reduce(O * dO, "... bq d -> ... bq 1", "sum")

    dQ, dK, dV = torch.zeros(Q.shape), torch.zeros(K.shape), torch.zeros(V.shape)
    for j in range(it_tk):
      col_s = j*bk
      Kj = K[..., col_s:col_s+bk, :]
      Vj = V[..., col_s:col_s+bk, :]
      for i in range(it_tq):
        s = i*bq
        dOi = dO[..., s:s+bq, :]
        Qi = Q[..., s:s+bq, :]
        Di = D[..., s:s+bq, :]
        Li = einops.rearrange(L[..., s:s+bq], "... -> ... 1")

        Sij = einops.einsum(Qi, Kj, "... bq d, ... bk d -> ... bq bk") / d_model_sqrt
        Pij = torch.exp(Sij - Li)
        dV[..., col_s:col_s+bk, :] += einops.einsum(Pij, dOi, "... bq bk, ... bq d -> ... bk d")
        dPij = einops.einsum(dOi, Vj, "... bq d, ... bk d -> ... bq bk")
        dSij = Pij * (dPij - Di) / d_model_sqrt
        dQ[..., s:s+bq, :] += einops.einsum(dSij, Kj, "... bq bk, ... bk d -> ... bq d")
        dK[..., col_s:col_s+bk, :] += einops.einsum(dSij, Qi, "... bq bk, ... bq d -> ... bk d")
    return dQ, dK, dV, None


  @staticmethod
  def backward(ctx, dO):
    # return FlashAttention2InPytorch._backward(ctx, dO)
    return FlashAttention2InPytorch._tiled_backward(ctx, dO)


if __name__ == "__main__":
  B, T, D = 1, 128, 64
  torch.random.manual_seed(1)
  Q, K, V, dO = [torch.randn((1, 2, T, D), requires_grad=True) for _ in range(4)]
  mask = torch.tril(torch.ones((T, T)))
  print(Q, K, V)
  impl = FlashAttention2InPytorch().apply
  impl(Q, K, V, False).backward(dO)
