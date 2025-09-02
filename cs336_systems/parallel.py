import os
from typing import Any, Type
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import timeit
from cs336_basics import module, trainer, opt, utils
import numpy as np

def setup(rank, world_size):
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = "29500"
  dist.init_process_group("gloo", rank=rank, world_size=world_size)

def distributed_demo(rank, world_size, tensor_size):
  setup = f"""
setup({rank}, {world_size})
data = torch.randn(({tensor_size},))
"""
  stmt = f"""
dist.all_reduce(data, op=torch.distributed.ReduceOp.MAX, async_op=False)
"""
  # setup(rank, world_size)
  # data = torch.randn((tensor_size,))
  # print(f"rank {rank} data (before all-reduce): {data}")
  # dist.all_reduce(data, op=torch.distributed.ReduceOp.MAX, async_op=False)
  # print(f"rank {rank} data (after all-reduce): {data}")
  cost = timeit.timeit(stmt=stmt, setup=setup, number=1, globals=globals())
  print(f"rank {rank} data {tensor_size} all-reduce cost: {cost}")


class GradHook():
  def __init__(self):
    self.handlers = []
  
  def hook(self, tensor):
    handler = dist.all_reduce(tensor.grad, op=torch.distributed.ReduceOp.SUM, async_op=True)
    self.handlers.append(handler)

  def wait_until_finish(self):
    for handler in self.handlers:
      handler.wait()
    self.handlers.clear()


class NaiveDDP(nn.Module):
  def __init__(self, module: nn.Module):
    super().__init__()
    
    self.module = module
    self.world_size = dist.get_world_size()
    self.hook = GradHook()
    for param in self.module.parameters():
      dist.broadcast(param, src=0)
      if param.requires_grad:
        param.register_post_accumulate_grad_hook(self.hook.hook)

  def forward(self, x):
    return self.module(x)
  
  def finish_gradient_synchronization(self, opt):
    self.hook.wait_until_finish()
    for group in opt.param_groups:
      for p in group["params"]:
        if p.grad is None:
          continue
        p.grad.data /= self.world_size

    # for group in opt.param_groups:
    #   grads = [p.grad.data for p in group["params"] if p.grad is not None]
    #   flat = torch._utils._flatten_dense_tensors(grads)
    #   dist.all_reduce(flat, op=torch.distributed.ReduceOp.SUM)
    #   flat /= self.world_size
    #   for g, reduced in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
    #     g.copy_(reduced)


class BucketedGradHooks():
  def __init__(self, bucket_size):
    self.handlers = []
    self.bucket_size_mb = bucket_size
    self.buckets = [[]]
    self.flats = []
    self.current_bucket_size = 0

  def _flush(self):
    if self.current_bucket_size == 0:
      return
    flat = torch._utils._flatten_dense_tensors(self.buckets[-1])
    self.flats.append(flat)
    handler = dist.all_reduce(flat, op=torch.distributed.ReduceOp.SUM, async_op=True)
    
    self.handlers.append(handler)
    self.buckets.append([])
    self.current_bucket_size = 0

  def hook(self, tensor: torch.Tensor):
    self.buckets[-1].append(tensor.grad.data)
    self.current_bucket_size += tensor.numel()
    if self.current_bucket_size > self.bucket_size_mb:
      self._flush()

  def wait_until_finish(self):
    self._flush()
    for handler in self.handlers:
      handler.wait()

    for bucket, flat in zip(self.buckets[:-1], self.flats):
      for g, reduced in zip(bucket, torch._utils._unflatten_dense_tensors(flat, bucket)):
        g.copy_(reduced)
    
    self.buckets = [[]]
    self.current_bucket_size = 0
    self.flats.clear()
    self.handlers.clear()


class OverLapDDPBucketed(nn.Module):
  def __init__(self, module: nn.Module, bucket_size_mb: float):
    super().__init__()
    self.module = module
    self.world_size = dist.get_world_size()
    self.hook = BucketedGradHooks(bucket_size_mb)
    for param in self.module.parameters():
      dist.broadcast(param, src=0)
      if param.requires_grad:
        param.register_post_accumulate_grad_hook(self.hook.hook)
  
  def forward(self, x):
    return self.module(x)
  
  def finish_gradient_synchronization(self, opt):
    self.hook.wait_until_finish()
    for group in opt.param_groups:
      for p in group["params"]:
        if p.grad is None:
          continue
        p.grad.data /= self.world_size


class ShardingOptimizer(torch.optim.Optimizer):
  def __init__(self, params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs: Any):
    self.cls = optimizer_cls
    self.instance_kwargs = kwargs
    self.rank = dist.get_rank()
    self.world_size = dist.get_world_size()
    super().__init__(params, kwargs)

  def step(self, closure=None):
    loss = self.local_opt.step(closure)
    for group in self.param_groups:
      for i, param in enumerate(group["params"]):
        dist.broadcast(param, src=i % self.world_size)
    return loss

  def add_param_group(self, param_group: dict[str, Any]):
    local_params = [p for i, p in enumerate(param_group["params"]) if i % self.world_size == self.rank]
    self.local_opt = self.cls(local_params, **self.instance_kwargs)
    super().add_param_group(param_group)


def _benchmark_reduce_communication(rank: int, world_size: int):
  device = "cpu"

  def prepare_data():
    np.random.seed(1024)
    data = np.random.randint(0, 10000, 65)
    return trainer.get_batch(data, 20, 64, device=device)

  setup(rank, world_size)
  model = NaiveDDP(module.Transformer(10000, 64, 1024, 24, 16, 4096, 10000, device=device))
  for param in model.parameters():
      dist.broadcast(param, src=0)
  optimizer = opt.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2, eps=1e-8, betas=(0.9, 0.999))
  inputs, targets = prepare_data()
  batch = len(inputs) // world_size
  local_inputs = inputs[batch*rank:batch*rank+batch, :]
  local_targets = targets[batch*rank:batch*rank+batch, :]

  def one_round():
    optimizer.zero_grad()
    outputs = model(local_inputs)
    loss = utils.cross_entropy_loss(outputs, local_targets)
    loss.backward()
    model.finish_gradient_synchronization(optimizer)
    optimizer.step()

  # warmup
  for _ in range(5):
    one_round()
  
  import time
  start = time.time()
  for _ in range(5):
    one_round()
  print(f"rank {rank}, cost { time.time() - start }")


def benchmark_reduce_communication(world_size=2):
  # stmt = f"""mp.spawn(fn=_benchmark_reduce_communication, args=({world_size}, False), nprocs={world_size}, join=True)"""
  # cost = timeit.timeit(stmt=stmt, number=1, globals=globals())
  # print("no zip cost", cost)
  stmt = f"""mp.spawn(fn=_benchmark_reduce_communication, args=({world_size}, ), nprocs={world_size}, join=True)"""
  cost = timeit.timeit(stmt=stmt, number=1, globals=globals())
  print("overlap cost", cost)
  # mp.spawn(fn=_benchmark_reduce_communication, args=(world_size, False), nprocs=world_size, join=True)


if __name__ == "__main__":
  # world_sizes = [2, 4, 6]
  # tensor_sizes = [1000000, 10000000, 100000000, 1000000000]
  # for world_size in world_sizes:
  #   for tensor_size in tensor_sizes:
  #     print(f"========= world size {world_size}, tensor size {tensor_size} =========")
  #     mp.spawn(fn=distributed_demo, args=(world_size, tensor_size), nprocs=world_size, join=True)
  benchmark_reduce_communication()