from cs336_basics import module
import torch
import pandas
import timeit


def one_step(model, inputs, targets, forward_only):
  outputs = model(*inputs)
  if not forward_only:
    loss = torch.nn.L1Loss()
    outputs.backward()
  torch.mps.synchronize


def test_one(batch_size, context_length, d_model, num_heads, forward_only, device, queue=None):
  warmups = 5
  n_steps = 100
  setup = f"""
inputs = torch.randn(({batch_size}, {context_length}, {d_model}), device="{device}")
targets = torch.randn(({batch_size}, {context_length}, {d_model}), device="{device}")
token_positions = torch.arange({context_length})
model = module.MultiHeadSelfAttention({d_model}, {num_heads}, {context_length}, theta=10000, device="{device}")
[one_step(model, (inputs, token_positions), targets, {forward_only}) for _ in range({warmups})]
"""
  stmt = f"""
[one_step(model, (inputs, token_positions), targets, {forward_only}) for _ in range({n_steps})]
"""
  cost = timeit.timeit(stmt=stmt, setup=setup, number=1, globals=globals())
  if queue is None:
    return cost
  else:
    queue.put(cost)


def test_naive_attn():
  device = "mps"

  batch_size = 4
  context_length = (256, 1024, 4096, 8192, 16384)
  d_model = 1024
  num_heads = (64, 32, 16, 8, 1)

  results = []
  for seq_len in context_length:
    for n_head in num_heads:
      for forward_only in (True, False):
        cost = test_one(batch_size, seq_len, d_model, n_head, forward_only, device=device)
        results.append((seq_len, n_head, forward_only, cost))
  title = ("seq_len", "n_head", "forward_only", "cost")
  table = {}
  for x in zip(title, *results):
    table[x[0]] = x[1:]
  md = pandas.DataFrame(data=table).to_markdown()
  print(md)


if __name__ == "__main__":
  test_naive_attn()