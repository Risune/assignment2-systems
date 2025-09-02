from cs336_basics import trainer, module, tokenizer, utils, opt
import numpy as np
import timeit
import pandas
import torch
import multiprocessing


def set_up(batch_size, vocab_size, context_length, d_m, n_l, n_h, d_f, rope_theta, device):
  data = np.random.randint(0, 10000, 100000)
  model = module.Transformer(vocab_size, context_length, d_m, n_l, n_h, d_f, rope_theta, device=device)
  inputs, targets = trainer.get_batch(data, batch_size, context_length, device)
  optimizer = opt.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2, eps=1e-8, betas=(0.9, 0.999))
  return model, inputs, targets, optimizer


def one_step(model, inputs, targets, optimizer, forward_only, device, dtype):
  def _run():
    outputs = model(inputs)
    if not forward_only:
      loss = utils.cross_entropy_loss(outputs, targets)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    torch.mps.synchronize
  
  if dtype == torch.float32:
    _run()
  else:
    with torch.autocast(device_type=device, dtype=dtype):
      _run()


def init_model_and_run(queue, batch_size, vocab_size, context_length, d_m, n_l, n_h, d_f, rope_theta, warmups, n_steps, forward_only, device, dtype, model_compile=False):
  print("testing:", batch_size, vocab_size, context_length, d_m, n_l, n_h, d_f, warmups, n_steps, forward_only)
  setup = f"""
model, inputs, targets, optimizer = set_up({batch_size}, {vocab_size}, {context_length}, {d_m}, {n_l}, {n_h}, {d_f}, {rope_theta}, "{device}")
if {model_compile}:
  torch.compile(model)
[one_step(model, inputs, targets, optimizer, {forward_only}, "{device}", torch.{dtype}) for _ in range({warmups})]
"""
  stmt = f"""
[one_step(model, inputs, targets, optimizer, {forward_only}, "{device}", torch.{dtype}) for _ in range({n_steps})]
"""
  cost = timeit.timeit(stmt=stmt, setup=setup, number=1, globals=globals())

  if queue is not None:
    queue.put(cost)
    queue.close()
  else:
    return cost

def profile_time():
  device = "mps"

  vocab_size = 10000
  batch_size = 4
  context_length = 256
  d_model = (768, 1024)
  num_layers = (12, 24)
  num_heads = (12, 16)
  d_ff = (3072, 4096)
  rope_theta = 10000
  
  results = []
  for d_m, n_l, n_h, d_f in zip(d_model, num_layers, num_heads, d_ff):
    for warmups in (0, 1, 2, 5):
      for forward_only in (True, False):
        queue = multiprocessing.Queue()
        p = multiprocessing.Process(
          target=init_model_and_run, 
          args=(queue, batch_size, vocab_size, context_length, d_m, n_l, n_h, d_f, rope_theta, warmups, 10, forward_only, device, "float16")
        )
        p.start()
        p.join()
        cost = queue.get()
        results.append((
          d_m, n_l, n_h, d_f, warmups, forward_only, "%.4f" % cost
        ))
  title = ("d_model", "num_layers", "num_heads", "d_ff", "warmups", "forward_only", "cost")
  table = {}
  for x in zip(title, *results):
    table[x[0]] = x[1:]
  md = pandas.DataFrame(data=table).to_markdown()
  print(md)


def one_step_with_mem_profile(model, inputs, targets, optimizer, forward_only):
  start_mem = torch.mps.current_allocated_memory()
  outputs = model(inputs)
  if not forward_only:
    loss = utils.cross_entropy_loss(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  torch.mps.synchronize
  return torch.mps.current_allocated_memory() - start_mem

def profile_mem():
  device = "mps"

  vocab_size = 10000
  batch_size = 4
  context_length = (128, 256, 384)
  d_model = 1024
  num_layers = 24
  num_heads = 16
  d_ff = 4096
  rope_theta = 10000

  results = []
  for seq_len in context_length:
    model, inputs, targets, optimizer = set_up(batch_size, vocab_size, seq_len, d_model, num_layers, num_heads, d_ff, rope_theta, device)
    for forward_only in (True, False):
      print("testing: ", seq_len, forward_only)
      cost = one_step_with_mem_profile(model, inputs, targets, optimizer, forward_only)
      print("testing done: ", seq_len, forward_only, cost)
      results.append((seq_len, forward_only, cost))

  title = ("seq_len", "forward_only", "cost")
  table = {}
  for x in zip(title, *results):
    table[x[0]] = x[1:]
  md = pandas.DataFrame(data=table).to_markdown()
  print(md)


def profile_attn_time():
  device = "mps"

  vocab_size = 10000
  batch_size = 4
  context_length = (256, 1024)
  d_model = 1024
  num_layers = 4
  num_heads = (64, 32, 16, 8, 1)
  d_ff = 4096
  rope_theta = 10000
  
  warmups = 5
  n_step = 10

  results = []
  for seq_len in context_length:
    for num_head in num_heads:
      for forward_only in (True, False):
        queue = multiprocessing.Queue()
        p = multiprocessing.Process(
          target=init_model_and_run, 
          args=(queue, batch_size, vocab_size, seq_len, d_model, num_layers, num_head, d_ff, rope_theta, warmups, n_step, forward_only, device, "float32", True)
        )
        p.start()
        p.join(60)
        if p.is_alive():
          print("timeout!!")
          cost = -1
          p.kill()
        else:
          cost = queue.get()
        results.append((
          seq_len, num_head, forward_only, "%.4f" % cost
        ))
  title = ("seq_len", "num_heads", "forward_only", "cost")
  table = {}
  for x in zip(title, *results):
    table[x[0]] = x[1:]
  md = pandas.DataFrame(data=table).to_markdown()
  print(md)


def profile_attn_mem():
  device = "mps"

  vocab_size = 10000
  batch_size = 4
  context_length = (256, 1024)
  d_model = 1024
  num_layers = 4
  num_heads = (64, 32, 16, 8, 1)
  d_ff = 4096
  rope_theta = 10000

  results = []
  for seq_len in context_length:
    for num_head in num_heads:
      model, inputs, targets, optimizer = set_up(batch_size, vocab_size, seq_len, d_model, num_layers, num_head, d_ff, rope_theta, device)
      for forward_only in (True, False):
        print("testing: ", seq_len, forward_only)
        cost = one_step_with_mem_profile(model, inputs, targets, optimizer, forward_only)
        print("testing done: ", seq_len, forward_only, cost)
        results.append((seq_len, num_head, forward_only, cost))

  title = ("seq_len", "num_head", "forward_only", "cost")
  table = {}
  for x in zip(title, *results):
    table[x[0]] = x[1:]
  md = pandas.DataFrame(data=table).to_markdown()
  print(md)


if __name__ == "__main__":
  # profile_time()
  # profile_attn_time()
  # profile_attn_mem()
  ...
