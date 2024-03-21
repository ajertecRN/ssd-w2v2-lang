from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pandas as pd


@dataclass
class AssembleSplitsArguments:
    DEFAULT_SPLITS_COLUMN: ClassVar[str] = "split"

    recordings_path: Path
    labels_path: Path
    result_dir: Path
    splits_column: str = DEFAULT_SPLITS_COLUMN

    def __post_init__(self):
        self.recordings_path = Path(self.recordings_path).resolve()
        self.labels_path = Path(self.labels_path).resolve()
        self.result_dir = Path(self.result_dir).resolve()
        self.splits_column = str(
            self.DEFAULT_SPLITS_COLUMN
            if self.splits_column is None
            else self.splits_column
        )

        assert self.recordings_path.exists(), f"Recordings path: Expected it to exist"
        assert self.labels_path.exists(), f"Labels path: Expected it to exist"
        assert self.splits_column, f"Splits column: Expected a value, got nil"

    @staticmethod
    def get_parser():
        parser = ArgumentParser()

        inputs = parser.add_argument_group("Inputs")
        outputs = parser.add_argument_group("Outputs")

        inputs.add_argument(
            "--recordings_path",
            type=str,
            required=True,
            metavar="FILEPATH",
            help="Path to the recordings CSV file (needs name and split column)",
        )
        inputs.add_argument(
            "--splits_column",
            type=str,
            default=AssembleSplitsArguments.DEFAULT_SPLITS_COLUMN,
            metavar="STRING",
            help=(
                "Name of the column where splits are located. Default: "
                f"{AssembleSplitsArguments.DEFAULT_SPLITS_COLUMN}"
            ),
        )
        inputs.add_argument(
            "--labels_path",
            type=str,
            required=True,
            metavar="FILEPATH",
            help="Path to the labels CSV file",
        )

        outputs.add_argument(
            "--result_dir",
            type=str,
            required=True,
            metavar="DIR",
            help="Path to the directory where splits will be assembled to",
        )

        return parser

    @staticmethod
    def from_args(args) -> "AssembleSplitsArguments":
        recordings_path = args.recordings_path
        labels_path = args.labels_path
        result_dir = args.result_dir
        splits_column = args.splits_column

        return AssembleSplitsArguments(
            recordings_path=recordings_path,
            labels_path=labels_path,
            result_dir=result_dir,
            splits_column=splits_column,
        )


def assemble_splits(arguments: AssembleSplitsArguments):
    recordings_df = pd.read_csv(
        arguments.recordings_path, usecols=["name", arguments.splits_column]
    )
    labels_df = pd.read_csv(arguments.labels_path)
    labels_df["name"] = labels_df["audio_relative_path"].apply(lambda x: Path(x).name)

    merged_df = labels_df.merge(
        recordings_df, how="left", left_on="name", right_on="name"
    ).dropna(axis=0, how="any")
    splits = sorted(merged_df[arguments.splits_column].unique())

    arguments.result_dir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        df = merged_df[merged_df[arguments.splits_column] == split].drop(
            ["name", arguments.splits_column], axis=1
        )
        df = df.drop_duplicates().sort_values(list(df.columns)).reset_index(drop=True)

        df.to_csv(arguments.result_dir / f"{split}.csv", index=False)


def main():
    parser = AssembleSplitsArguments.get_parser()
    args = parser.parse_args()
    arguments = AssembleSplitsArguments.from_args(args=args)

    assemble_splits(arguments=arguments)


if __name__ == "__main__":
    main()
