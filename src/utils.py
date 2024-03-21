import os
import json
import yaml
from pathlib import Path

import numpy as np
import scipy

from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS


def get_optimizer_params(model, weight_decay: float):
    decay_parameters = get_parameter_names(
        model, forbidden_layer_types=ALL_LAYERNORM_LAYERS
    )
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters


def minmax_scale(arr: np.ndarray, v_min: float, v_max: float, dtype):
    arr_min = np.min(arr)
    arr_max = np.max(arr)

    scaled = (arr - arr_min) * ((v_max - v_min) / (arr_max - arr_min)) + v_min

    return scaled.astype(dtype)


def img_y_log_scale(img, target_x, target_y, base):
    y_max, x_max = img.shape

    lin_space_x = np.linspace(0, x_max - 1, x_max, dtype=np.uint16)
    lin_space_y = np.linspace(0, y_max - 1, y_max, dtype=np.uint16)

    lin_space_target = np.linspace(0, x_max - 1, target_x)
    log_space = np.logspace(0, np.log(y_max) / np.log(base), target_y, base=base)

    interpolator = scipy.interpolate.RectBivariateSpline(lin_space_y, lin_space_x, img)

    return interpolator(log_space, lin_space_target)


def load_yaml(filepath: str):
    with open(filepath, "r", encoding="utf-8") as infile:
        return yaml.safe_load(infile)


def save_yaml(filepath: str, data: dict):
    with open(filepath, "w", encoding="utf8") as outfile:
        yaml.dump(data, outfile, default_flow_style=False, allow_unicode=True)


def save_json(filepath: str, data: dict):
    with open(filepath, "w", encoding="utf-8") as outfile:
        json.dump(data, outfile, indent=4)


def save_args(args, output_dir: str, output_filename: str = "training_args.json"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_json(filepath=os.path.join(output_dir, output_filename), data=vars(args))
