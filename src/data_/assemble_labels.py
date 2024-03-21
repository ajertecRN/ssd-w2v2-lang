import argparse
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Optional

import pandas as pd


@dataclass
class AssembleLabelsArguments:
    DEFAULT_HUMAN_DATASET_DIR: ClassVar[Optional[Path]] = None
    DEFAULT_SYNTHETIC_DATASET_DIR: ClassVar[Optional[Path]] = None
    DEFAULT_ROOT_DIR: ClassVar[Path] = Path("/").resolve()
    DEFAULT_HUMAN_AUDIO_NAME: ClassVar[str] = "standardized"
    DEFAULT_SYNTHETIC_AUDIO_NAME: ClassVar[str] = "original"
    DEFAULT_HUMAN_LABEL: ClassVar[str] = "human"
    DEFAULT_SYNTHETIC_LABEL: ClassVar[str] = "robot"

    result_path: Path
    human_dataset_dir: Optional[Path] = DEFAULT_HUMAN_DATASET_DIR
    synthetic_dataset_dir: Optional[Path] = DEFAULT_SYNTHETIC_DATASET_DIR
    root_dir: Path = DEFAULT_ROOT_DIR
    human_audio_name: str = DEFAULT_HUMAN_AUDIO_NAME
    synthetic_audio_name: str = DEFAULT_SYNTHETIC_AUDIO_NAME
    human_label: str = DEFAULT_HUMAN_LABEL
    synthetic_label: str = DEFAULT_SYNTHETIC_LABEL

    def __post_init__(self):
        self.result_path = Path(self.result_path).resolve()
        self.human_dataset_dir = (
            self.DEFAULT_HUMAN_DATASET_DIR
            if self.human_dataset_dir is None
            else Path(self.human_dataset_dir).resolve()
        )
        self.synthetic_dataset_dir = (
            self.DEFAULT_SYNTHETIC_DATASET_DIR
            if self.synthetic_dataset_dir is None
            else Path(self.synthetic_dataset_dir).resolve()
        )
        self.root_dir = Path(
            self.DEFAULT_ROOT_DIR if self.root_dir is None else self.root_dir
        )
        self.human_audio_name = str(
            self.DEFAULT_HUMAN_AUDIO_NAME
            if self.human_audio_name is None
            else self.human_audio_name
        )
        self.synthetic_audio_name = str(
            self.DEFAULT_SYNTHETIC_AUDIO_NAME
            if self.synthetic_audio_name is None
            else self.synthetic_audio_name
        )
        self.human_label = str(
            self.DEFAULT_HUMAN_LABEL if self.human_label is None else self.human_label
        ).strip()
        self.synthetic_label = str(
            self.DEFAULT_SYNTHETIC_LABEL
            if self.synthetic_label is None
            else self.synthetic_label
        ).strip()

        assert self.human_dataset_dir or self.synthetic_dataset_dir, (
            "Human dataset directory, Synthetic dataset directory: Expected at least "
            "one to be defined"
        )
        assert (
            self.human_dataset_dir is None or self.human_dataset_dir.exists()
        ), "Human dataset directory: Expected it to exist"
        assert (
            self.synthetic_dataset_dir is None or self.synthetic_dataset_dir.exists()
        ), "Synthetic dataset directory: Expected it to exist"
        assert self.root_dir.exists(), "Root directory: Expected it to exist"
        assert self.human_audio_name, "Human audio name: Expected a value, got nil"
        assert (
            self.synthetic_audio_name
        ), "Synthetic audio name: Expected a value, got nil"
        assert self.human_label, "Human label: Expected a value, got nil"
        assert self.synthetic_label, "Synthetic label: Expected a value, got nil"
        assert (
            self.human_label != self.synthetic_label
        ), "Human label, Synthetic label: Expected them to be different"

    @staticmethod
    def get_parser():
        parser = argparse.ArgumentParser()

        inputs = parser.add_argument_group("Inputs")
        outputs = parser.add_argument_group("Outputs")

        inputs.add_argument(
            "--human_dataset_dir",
            type=str,
            default=AssembleLabelsArguments.DEFAULT_HUMAN_DATASET_DIR,
            metavar="DIR",
            help=(
                "Path to the human dataset root directory. Default: ignore human "
                "samples"
            ),
        )
        inputs.add_argument(
            "--synthetic_dataset_dir",
            type=str,
            default=AssembleLabelsArguments.DEFAULT_SYNTHETIC_DATASET_DIR,
            metavar="DIR",
            help=(
                "Path to the synthetic dataset root directory. Default: ignore "
                "synthetic samples"
            ),
        )
        inputs.add_argument(
            "--root_dir",
            type=str,
            default=AssembleLabelsArguments.DEFAULT_ROOT_DIR,
            metavar="DIR",
            help=(
                "Path to the directory relative to which the audio is pathed. Default: "
                f"{AssembleLabelsArguments.DEFAULT_ROOT_DIR}"
            ),
        )
        inputs.add_argument(
            "--human_audio_name",
            type=str,
            default=AssembleLabelsArguments.DEFAULT_HUMAN_AUDIO_NAME,
            metavar="DIRNAME",
            help=(
                "Name of the directory containing human audio. Default: "
                f"{AssembleLabelsArguments.DEFAULT_HUMAN_AUDIO_NAME}"
            ),
        )
        inputs.add_argument(
            "--synthetic_audio_name",
            type=str,
            default=AssembleLabelsArguments.DEFAULT_SYNTHETIC_AUDIO_NAME,
            metavar="DIRNAME",
            help=(
                "Name of the directory containing synthetic audio. Default: "
                f"{AssembleLabelsArguments.DEFAULT_SYNTHETIC_AUDIO_NAME}"
            ),
        )

        outputs.add_argument(
            "--human_label",
            type=str,
            default=AssembleLabelsArguments.DEFAULT_HUMAN_LABEL,
            metavar="STRING",
            help=(
                "Name of the label used for human audio. Default: "
                f"{AssembleLabelsArguments.DEFAULT_HUMAN_LABEL}"
            ),
        )
        outputs.add_argument(
            "--synthetic_label",
            type=str,
            default=AssembleLabelsArguments.DEFAULT_SYNTHETIC_LABEL,
            metavar="STRING",
            help=(
                "Name of the label used for synthetic audio. Default: "
                f"{AssembleLabelsArguments.DEFAULT_SYNTHETIC_LABEL}"
            ),
        )
        outputs.add_argument(
            "--result_path",
            type=str,
            required=True,
            metavar="FILEPATH",
            help="Path to the result CSV",
        )

        return parser

    @staticmethod
    def from_args(args) -> "AssembleLabelsArguments":
        result_path = args.result_path
        human_dataset_dir = args.human_dataset_dir
        synthetic_dataset_dir = args.synthetic_dataset_dir
        root_dir = args.root_dir
        human_audio_name = args.human_audio_name
        synthetic_audio_name = args.synthetic_audio_name
        human_label = args.human_label
        synthetic_label = args.synthetic_label

        return AssembleLabelsArguments(
            result_path=result_path,
            human_dataset_dir=human_dataset_dir,
            synthetic_dataset_dir=synthetic_dataset_dir,
            root_dir=root_dir,
            human_audio_name=human_audio_name,
            synthetic_audio_name=synthetic_audio_name,
            human_label=human_label,
            synthetic_label=synthetic_label,
        )


def get_relative_audio_paths(recordings_path: Path, audio_dir: Path, root_dir: Path):
    filenames = set(pd.read_csv(recordings_path, usecols=["name"])["name"].unique())

    audio_paths = glob.glob(f"{audio_dir}/*")
    audio_paths = [Path(audio_path).resolve() for audio_path in audio_paths]
    audio_paths = [audio_path for audio_path in audio_paths if audio_path.is_file()]

    audio_paths_to_take = [
        audio_path for audio_path in audio_paths if audio_path.name in filenames
    ]
    relative_audio_paths_to_take = [
        audio_path.relative_to(root_dir) for audio_path in audio_paths_to_take
    ]

    return relative_audio_paths_to_take


def assemble_labels(arguments: AssembleLabelsArguments):
    chunks = list()

    if arguments.human_dataset_dir:
        relative_audio_paths_to_take = get_relative_audio_paths(
            recordings_path=arguments.human_dataset_dir / "data" / "recordings.csv",
            audio_dir=arguments.human_dataset_dir
            / "audio"
            / arguments.human_audio_name,
            root_dir=arguments.root_dir,
        )
        labels = pd.DataFrame(
            relative_audio_paths_to_take, columns=["audio_relative_path"]
        )
        labels["label"] = arguments.human_label

        chunks.append(labels)

    if arguments.synthetic_dataset_dir:
        relative_audio_paths_to_take = get_relative_audio_paths(
            recordings_path=arguments.synthetic_dataset_dir
            / "data"
            / "distribution.csv",
            audio_dir=arguments.synthetic_dataset_dir
            / "audio"
            / arguments.synthetic_audio_name,
            root_dir=arguments.root_dir,
        )
        labels = pd.DataFrame(
            relative_audio_paths_to_take, columns=["audio_relative_path"]
        )
        labels["label"] = arguments.synthetic_label

        chunks.append(labels)

    chunks = (
        pd.concat(chunks, axis=0)
        .dropna()
        .drop_duplicates()
        .sort_values(["audio_relative_path", "label"])
        .reset_index(drop=True)
    )

    arguments.result_path.parent.mkdir(parents=True, exist_ok=True)
    chunks.to_csv(arguments.result_path, index=False)


def main():
    parser = AssembleLabelsArguments.get_parser()
    args = parser.parse_args()
    arguments = AssembleLabelsArguments.from_args(args=args)

    assemble_labels(arguments=arguments)


if __name__ == "__main__":
    main()
