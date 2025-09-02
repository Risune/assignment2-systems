import torch
import torch.nn as nn


class ToyModel(nn.Module):
  def __init__(self, in_features, out_features):
    super().__init__()
    self.fc1 = nn.Linear(in_features, 10, bias=False)
    self.ln = nn.LayerNorm(10)
    self.fc2 = nn.Linear(10, out_features, bias=False)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.relu(self.fc1(x))
    print(x)
    x = self.ln(x)
    print(x)
    x = self.fc2(x)
    print(x)
    return x

if __name__ == "__main__":
  device = "mps"
  model = ToyModel(5, 3).to(device)
  x = torch.randn((1, 5)).to(device)
  targets = torch.Tensor([1]).to(device)
  with torch.autocast(device_type=device, dtype=torch.bfloat16):
    for params in model.parameters():
      print(params)
    print(x)
    y = model(x)
    loss = nn.functional.cross_entropy(y, targets)
    print("loss", loss)
    loss.backward()
    print("grad")
    for params in model.parameters():
      print(params.grad)

