import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, Optional

import numpy as np
import pandas as pd

from utils.distribution import (
    distribute_voices as distribute_voices_prime,
    parse_vendor_to_probability,
)


@dataclass
class DistributeVoicesArguments:
    DEFAULT_RECORDINGS_NAME: ClassVar[str] = "recordings.csv"
    DEFAULT_VOICES_NAME: ClassVar[str] = "voices.csv"
    DEFAULT_RESULT_NAME: ClassVar[str] = "distribution.csv"
    DEFAULT_VENDOR_TO_PROBABILITY: ClassVar[Optional[Dict[str, float]]] = None
    DEFAULT_SEED: ClassVar[int] = 0
    DEFAULT_DROP_GENDER_MISMATCH: ClassVar[bool] = False
    DEFAULT_DROP_UNASSIGNED: ClassVar[bool] = False
    DEFAULT_VERBOSE: ClassVar[bool] = False

    dataset_dir: Path
    result_dir: Path
    recordings_name: str = DEFAULT_RECORDINGS_NAME
    voices_name: str = DEFAULT_VOICES_NAME
    result_name: str = DEFAULT_RESULT_NAME
    vendor_to_probability: Optional[Dict[str, float]] = DEFAULT_VENDOR_TO_PROBABILITY
    seed: int = DEFAULT_SEED
    drop_gender_mismatch: bool = DEFAULT_DROP_GENDER_MISMATCH
    drop_unassigned: bool = DEFAULT_DROP_UNASSIGNED
    verbose: bool = DEFAULT_VERBOSE

    def __post_init__(self):
        self.dataset_dir = Path(self.dataset_dir).resolve()
        self.result_dir = Path(self.result_dir).resolve()
        self.recordings_name = str(
            self.DEFAULT_RECORDINGS_NAME
            if self.recordings_name is None
            else self.recordings_name
        ).strip()
        self.voices_name = str(
            self.DEFAULT_VOICES_NAME if self.voices_name is None else self.voices_name
        ).strip()
        self.result_name = (
            self.DEFAULT_RESULT_NAME if self.result_name is None else self.result_name
        )
        self.vendor_to_probability = (
            self.DEFAULT_VENDOR_TO_PROBABILITY
            if self.vendor_to_probability is None
            else dict(self.vendor_to_probability)
        )
        self.seed = int(self.DEFAULT_SEED if self.seed is None else self.seed)
        self.drop_gender_mismatch = bool(
            self.DEFAULT_DROP_GENDER_MISMATCH
            if self.drop_gender_mismatch is None
            else self.drop_gender_mismatch
        )
        self.drop_unassigned = bool(
            self.DEFAULT_DROP_UNASSIGNED
            if self.drop_unassigned is None
            else self.drop_unassigned
        )
        self.verbose = bool(
            self.DEFAULT_VERBOSE if self.verbose is None else self.verbose
        )

        assert (
            self.dataset_dir.exists()
        ), f"Dataset directory: {self.dataset_dir} doesn't exist"
        assert (
            self.result_dir.exists()
        ), f"Result directory: {self.result_dir} doesn't exist"
        assert self.recordings_name, f"Recordings name: Expected a value, got nil"
        assert self.voices_name, f"Voices name: Expected a value, got nil"
        assert self.result_name, f"Result name: Expected a value, got nil"
        assert (
            self.recordings_path.exists()
        ), f"Recordings path: {self.recordings_path} doesn't exist"
        assert (
            self.voices_path.exists()
        ), f"Voices path: {self.voices_path} doesn't exist"
        assert (
            0 >= self.seed < 2**32
        ), f"Seed: Expected to be in [0, 2^32], got {self.seed}"

    # region Properties
    @property
    def recordings_path(self) -> Path:
        return self.dataset_dir / self.recordings_name

    @property
    def voices_path(self) -> Path:
        return self.result_dir / self.voices_name

    @property
    def result_path(self) -> Path:
        return self.result_dir / self.result_name

    # endregion

    @staticmethod
    def get_parser():
        parser = argparse.ArgumentParser()

        inputs = parser.add_argument_group("Inputs")
        outputs = parser.add_argument_group("Outputs")

        inputs.add_argument(
            "--dataset_dir",
            type=str,
            required=True,
            metavar="DIR",
            help="Path to the directory of the dataset",
        )
        inputs.add_argument(
            "--recordings_name",
            type=str,
            default=DistributeVoicesArguments.DEFAULT_RECORDINGS_NAME,
            metavar="FILENAME",
            help=(
                "Name of the recordings CSV. Default: "
                f"{DistributeVoicesArguments.DEFAULT_RECORDINGS_NAME}"
            ),
        )
        inputs.add_argument(
            "--voices_name",
            type=str,
            default=DistributeVoicesArguments.DEFAULT_VOICES_NAME,
            metavar="FILENAME",
            help=(
                "Name of the voices CSV. Default: "
                f"{DistributeVoicesArguments.DEFAULT_VOICES_NAME}"
            ),
        )

        outputs.add_argument(
            "--result_dir",
            type=str,
            required=True,
            metavar="DIR",
            help="Path to the result directory.",
        )
        outputs.add_argument(
            "--result_name",
            type=str,
            default=DistributeVoicesArguments.DEFAULT_RESULT_NAME,
            metavar="FILENAME",
            help=(
                "Name of the result CSV. Default: "
                f"{DistributeVoicesArguments.DEFAULT_RESULT_NAME}"
            ),
        )
        outputs.add_argument(
            "--vendor_to_probability",
            type=str,
            default=DistributeVoicesArguments.DEFAULT_VENDOR_TO_PROBABILITY,
            metavar="STRING:FLOAT",
            help=(
                "List of vendor to probability mappings. Default: "
                "uniform distribution"
            ),
        )
        outputs.add_argument(
            "--seed",
            type=int,
            default=DistributeVoicesArguments.DEFAULT_SEED,
            metavar="UINT32",
            help=(
                "The seed for RNG operations. Default: "
                f"{DistributeVoicesArguments.DEFAULT_SEED}"
            ),
        )
        outputs.add_argument(
            "--drop_gender_mismatch",
            action="store_true",
            help=(
                "If set, will drop samples assigned voices with a non-matching gender"
            ),
        )
        outputs.add_argument(
            "--drop_unassigned",
            action="store_true",
            help="If set, will drop samples which were not assigned a voice",
        )
        outputs.add_argument(
            "--verbose",
            action="store_true",
            help="If set, will print out progress, otherwise only errors and warning",
        )

        return parser

    @staticmethod
    def from_args(args) -> "DistributeVoicesArguments":
        dataset_dir = args.dataset_dir
        result_dir = args.result_dir
        recordings_name = args.recordings_name
        voices_name = args.voices_name
        result_name = args.result_name
        vendor_to_probability = parse_vendor_to_probability(args.vendor_to_probability)
        seed = args.seed
        drop_gender_mismatch = args.drop_gender_mismatch
        drop_unassigned = args.drop_unassigned
        verbose = args.verbose

        return DistributeVoicesArguments(
            dataset_dir=dataset_dir,
            result_dir=result_dir,
            recordings_name=recordings_name,
            voices_name=voices_name,
            result_name=result_name,
            vendor_to_probability=vendor_to_probability,
            seed=seed,
            drop_gender_mismatch=drop_gender_mismatch,
            drop_unassigned=drop_unassigned,
            verbose=verbose,
        )


def distribute_voices(arguments: DistributeVoicesArguments):
    recordings_df = pd.read_csv(arguments.recordings_path)
    voices_df = pd.read_csv(arguments.voices_path)

    distribution = distribute_voices_prime(
        recordings_df=recordings_df,
        voices_df=voices_df,
        vendor_to_probability=arguments.vendor_to_probability,
        verbose=arguments.verbose,
    )
    if arguments.verbose:
        print(f"Distributed voices: {distribution.shape[0]} remaining")

    if arguments.drop_unassigned:
        distribution = distribution.dropna(
            axis=0,
            how="all",
            subset=["tts_vendor", "tts_name", "tts_engine"],
        )
        if arguments.verbose:
            print(
                f"Dropped samples without assigned voices: {distribution.shape[0]} "
                "remaining"
            )
    if arguments.drop_gender_mismatch:
        distribution = distribution.merge(
            recordings_df, how="left", left_on=["id", "name"], right_on=["id", "name"]
        )
        gender_mismatch = distribution["gender"] != distribution["tts_gender"]
        gender_mismatch_index = distribution[gender_mismatch].index

        distribution = distribution.drop(gender_mismatch_index).drop("gender", axis=1)
        if arguments.verbose:
            print(
                f"Dropped samples with gender mismatch: {distribution.shape[0]} "
                "remaining"
            )

    distribution = (
        distribution[["id", "name", "tts_vendor", "tts_name", "tts_engine"]]
        .sort_values(["id", "name", "tts_vendor", "tts_name", "tts_engine"])
        .reset_index(drop=True)
    )
    distribution.to_csv(arguments.result_path, index=False)


def main():
    parser = DistributeVoicesArguments.get_parser()
    args = parser.parse_args()
    arguments = DistributeVoicesArguments.from_args(args=args)

    pd.core.common.random_state(arguments.seed)
    np.random.seed(arguments.seed)

    distribute_voices(arguments=arguments)


if __name__ == "__main__":
    main()
