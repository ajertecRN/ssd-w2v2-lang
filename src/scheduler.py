from loguru import logger
from pprint import pformat
from torch.optim.lr_scheduler import (
    ExponentialLR,
    StepLR,
    ReduceLROnPlateau,
)

DEFAULT_GAMMA: float = 0.1

# polynomial:
DEFAULT_POWER: float = 1.0
DEFAULT_TOTAL_ITERS: int = 1

# step:
DEFAULT_STEP_SIZE: int = 1

# reduce on plateau:
DEFAULT_THRESHOLD: float = 1e-2
DEFAULT_THRESHOLD_MODE: str = "abs"
DEFAULT_MIN_LR: float = 1e-8
DEFAULT_EPS: float = 1e-8


def get_scheduler(
    scheduler_type: str,
    optimizer,
    **kwargs,
):
    logger.info(f"Scheduler type: '{scheduler_type}'; params: {pformat(kwargs)}")
    if scheduler_type == "exponential":
        gamma = kwargs.get("gamma", DEFAULT_GAMMA)
        return ExponentialLR(optimizer=optimizer, gamma=gamma, verbose=True)

    elif scheduler_type == "polynomial":
        # PolynomialLR missing from torch 1.12.1
        from torch.optim.lr_scheduler import PolynomialLR

        total_iters = kwargs.get("total_iters", DEFAULT_TOTAL_ITERS)
        power = kwargs.get("power", DEFAULT_POWER)
        return PolynomialLR(
            optimizer=optimizer, total_iters=total_iters, power=power, verbose=True
        )

    elif scheduler_type == "step":
        step_size = kwargs.get("step_size", DEFAULT_STEP_SIZE)
        gamma = kwargs.get("gamma", DEFAULT_GAMMA)
        return StepLR(
            optimizer=optimizer, step_size=step_size, gamma=gamma, verbose=True
        )

    elif scheduler_type == "reduce_on_plateau":
        threshold = kwargs.get("threshold", DEFAULT_THRESHOLD)
        threshold_mode = kwargs.get("threshold_mode", DEFAULT_THRESHOLD_MODE)
        min_lr = kwargs.get("min_lr", DEFAULT_MIN_LR)
        eps = kwargs.get("eps", DEFAULT_EPS)

        scheduler_mode = kwargs.get("mode")
        scheduler_factor = kwargs.get("factor")
        scheduler_patience = kwargs.get("patience")
        scheduler_cooldown = kwargs.get("cooldown")

        return ReduceLROnPlateau(
            optimizer=optimizer,
            mode=scheduler_mode,
            factor=scheduler_factor,
            patience=scheduler_patience,
            cooldown=scheduler_cooldown,
            threshold=threshold,
            threshold_mode=threshold_mode,
            min_lr=min_lr,
            eps=eps,
            verbose=True,
        )

    else:
        raise ValueError(f"Scheduler '{scheduler_type}' is not available.")
