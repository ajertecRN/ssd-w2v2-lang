import os
import torch
import pandas as pd
from pytorch_lightning.callbacks import BasePredictionWriter


class PredictionsWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval: str = "epoch"):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        torch.save(predictions, os.path.join(self.output_dir, "predictions.pt"))
