import warnings

import torch
import torchmetrics
from torch.nn import CrossEntropyLoss
import pytorch_lightning

from .utils import get_optimizer_params
from .scheduler import get_scheduler

SCHEDULER_INTERVAL = "epoch"
SCHEDULER_FREQUENCY = 1


class BaseLightning(pytorch_lightning.LightningModule):
    def __init__(
        self,
        model,
        input_args,
        num_classes,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-2,
        scheduler_monitor: str = "f1_dev",
        scheduler_kwargs: dict = {},
        scheduler_interval: str = SCHEDULER_INTERVAL,
        scheduler_frequency: int = SCHEDULER_FREQUENCY,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters(input_args)

        self.model = model
        self.num_classes = num_classes

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.scheduler_monitor = scheduler_monitor
        self.scheduler_interval = scheduler_interval
        self.scheduler_frequency = scheduler_frequency
        self.scheduler_kwargs = scheduler_kwargs

        self.training_metrics = torchmetrics.MetricCollection(
            {
                "accuracy": torchmetrics.Accuracy(
                    task="multiclass",
                    num_classes=self.num_classes,
                    average="micro",
                    validate_args=False,
                ),
                "f1_macro": torchmetrics.F1Score(
                    task="multiclass",
                    num_classes=self.num_classes,
                    average="macro",
                    validate_args=False,
                ),
                "precision_macro": torchmetrics.Precision(
                    task="multiclass",
                    num_classes=self.num_classes,
                    average="macro",
                    validate_args=False,
                ),
                "recall_macro": torchmetrics.Recall(
                    task="multiclass",
                    num_classes=self.num_classes,
                    average="macro",
                    validate_args=False,
                ),
                "f1_weighted": torchmetrics.F1Score(
                    task="multiclass",
                    num_classes=self.num_classes,
                    average="weighted",
                    validate_args=False,
                ),
            }
        )
        self.evaluation_metrics = torchmetrics.MetricCollection(
            {
                "confusion_matrix": torchmetrics.ConfusionMatrix(
                    task="multiclass",
                    num_classes=self.num_classes,
                    normalize=None,
                    validate_args=False,
                ),
                "accuracy": torchmetrics.Accuracy(
                    task="multiclass",
                    num_classes=self.num_classes,
                    average="micro",
                    validate_args=False,
                ),
                "f1_macro": torchmetrics.F1Score(
                    task="multiclass",
                    num_classes=self.num_classes,
                    average="macro",
                    validate_args=False,
                ),
                "precision_macro": torchmetrics.Precision(
                    task="multiclass",
                    num_classes=self.num_classes,
                    average="macro",
                    validate_args=False,
                ),
                "recall_macro": torchmetrics.Recall(
                    task="multiclass",
                    num_classes=self.num_classes,
                    average="macro",
                    validate_args=False,
                ),
                "f1_weighted": torchmetrics.F1Score(
                    task="multiclass",
                    num_classes=self.num_classes,
                    average="weighted",
                    validate_args=False,
                ),
            }
        )

    def _get_float_result_dict_wo_cm(self, results: dict, postfix: str) -> dict:
        return {
            f"{k}_{postfix}": float(v)
            for k, v in results.items()
            if k not in ["confusion_matrix"]
        }

    def forward(self, batch):
        raise NotImplementedError

    def predict_step(self, batch, _=None, __=None):
        raise NotImplementedError

    def training_step(self, batch, _):
        raise NotImplementedError

    def training_step_end(self, batch_parts):
        predictions = batch_parts["predictions"]
        targets = batch_parts["targets"]
        loss = batch_parts["loss"]

        if not isinstance(predictions, torch.Tensor):
            predictions = torch.stack(predictions)
        if not isinstance(targets, torch.Tensor):
            targets = torch.stack(targets)
        if not isinstance(loss, torch.Tensor):
            loss = torch.stack(loss)

        loss = loss.mean()

        binary_predictions = torch.argmax(predictions, dim=-1).float()
        binary_targets = targets.float()

        self.training_metrics(binary_predictions, binary_targets)
        results = self.training_metrics.compute()

        self.log("loss_train", loss.item())
        self.log_dict(self._get_float_result_dict_wo_cm(results, postfix="train"))

        train_scheduler_monitor = "_".join(self.scheduler_monitor.split("_")[:-1])
        self.log(
            train_scheduler_monitor,
            float(self.training_metrics[train_scheduler_monitor].compute()),
            prog_bar=True,
            logger=False,
        )

        return loss

    def training_epoch_end(self, _):
        self.training_metrics.reset()

    def validation_step(self, batch, _):
        return self.predict_step(batch=batch)

    def validation_step_end(self, batch_parts):
        predictions = batch_parts["predictions"]
        targets = batch_parts["targets"]

        if not isinstance(predictions, torch.Tensor):
            predictions = torch.stack(predictions)
        if not isinstance(targets, torch.Tensor):
            targets = torch.stack(targets)

        binary_predictions = torch.argmax(predictions, dim=-1).float()
        binary_targets = targets.float()

        self.evaluation_metrics(binary_predictions, binary_targets)

    def validation_epoch_end(self, _) -> None:
        if not self.trainer.sanity_checking:
            results = self.evaluation_metrics.compute()

            self.log_dict(
                {
                    name: value
                    for name, value in zip(
                        ("TN_dev", "FP_dev", "FN_dev", "TP_dev"),
                        results["confusion_matrix"].to(torch.float32).flatten(),
                    )
                }
            )
            self.log_dict(self._get_float_result_dict_wo_cm(results, postfix="dev"))

        self.evaluation_metrics.reset()

    def test_step(self, batch, _):
        return self.predict_step(batch=batch)

    def test_step_end(self, batch_parts):
        self.validation_step_end(batch_parts)

    def test_epoch_end(self, _):
        results = self.evaluation_metrics.compute()

        self.log_dict(
            {
                name: value
                for name, value in zip(
                    ("TN_test", "FP_test", "FN_test", "TP_test"),
                    results["confusion_matrix"].to(torch.float32).flatten(),
                )
            }
        )
        self.log_dict(self._get_float_result_dict_wo_cm(results, postfix="test"))

        self.evaluation_metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=get_optimizer_params(self.model, self.weight_decay),
            lr=self.learning_rate,
            weight_decay=0.0,
        )

        scheduler = get_scheduler(
            scheduler_type=self.scheduler_kwargs["type"],
            optimizer=optimizer,
            **self.scheduler_kwargs,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.scheduler_interval,
                "frequency": self.scheduler_frequency,
                "monitor": self.scheduler_monitor,
                "strict": True,
                "name": None,
            },
        }


class Wav2Vec2Lightning(BaseLightning):
    def __init__(
        self,
        model,
        input_args,
        *args,
        **kwargs,
    ):
        super().__init__(model=model, input_args=input_args, **kwargs)

    def forward(self, batch):
        return self.model(
            **{k: v for k, v in batch.items() if k not in ["filepath"]}
        ).logits

    def predict_step(self, batch, _=None, __=None):
        predictions = self.model(
            **{k: v for k, v in batch.items() if k not in ["filepath"]}
        ).logits

        return {
            "predictions": predictions,
            "targets": batch["labels"],
            "filepath": batch["filepath"],
        }

    def training_step(self, batch, _):
        outputs = self.model(
            **{k: v for k, v in batch.items() if k not in ["filepath"]}
        )

        return {
            "predictions": outputs.logits,
            "targets": batch["labels"],
            "loss": outputs.loss,
        }


class TimmLightning(BaseLightning):
    def __init__(
        self,
        model,
        input_args,
        *args,
        **kwargs,
    ):
        super().__init__(model=model, input_args=input_args, **kwargs)

        self.loss_fn = CrossEntropyLoss()

    def forward(self, batch):
        return self.model(batch["spectrograms"])

    def predict_step(self, batch, _=None, __=None):
        predictions = self.model(batch["spectrograms"])

        return {
            "filepath": batch["filepath"],
            "predictions": predictions,
            "targets": batch["labels"],
        }

    def training_step(self, batch, _):
        logits = self.model(batch["spectrograms"])
        loss = self.loss_fn(logits.view(-1, self.num_classes), batch["labels"].view(-1))

        return {
            "predictions": logits,
            "targets": batch["labels"],
            "loss": loss,
        }


def get_lightning_module_class(model_source: str):
    if model_source == "hft":
        module_class = Wav2Vec2Lightning

    elif model_source == "timm":
        module_class = TimmLightning
    else:
        raise ValueError

    return module_class


def get_lightning_module(
    model_source: str,
    model,
    input_args,
    num_classes: int,
    **kwargs,
):
    module_class = get_lightning_module_class(model_source)

    return module_class(
        model=model, input_args=input_args, num_classes=num_classes, **kwargs
    )
