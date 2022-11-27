import pandas as pd
import torch

b = torch.tensor([[1], [2], [2], [1]])

print(b.reshape((-1,)))