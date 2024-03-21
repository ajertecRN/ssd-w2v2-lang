import torch
from torch import Tensor


class ZeroMeanUnitVarNormalize(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor: Tensor) -> Tensor:
        return (tensor - tensor.mean()) / tensor.std()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
