import argparse
from dataclasses import dataclass
import glob
from pathlib import Path
import sys
from typing import ClassVar, List
import wave

from tqdm import tqdm


@dataclass
class PCMToWAVArguments:
    DEFAULT_OVERWRITE: ClassVar[bool] = False
    DEFAULT_VERBOSE: ClassVar[bool] = False

    source: str
    result_dir: Path
    overwrite: bool = DEFAULT_OVERWRITE
    verbose: bool = DEFAULT_VERBOSE

    def __post_init__(self):
        self.source = str(self.source)
        self.result_dir = Path(self.result_dir).resolve()
        self.overwrite = bool(
            self.DEFAULT_OVERWRITE if self.overwrite is None else self.overwrite
        )
        self.verbose = bool(
            self.DEFAULT_VERBOSE if self.verbose is None else self.verbose
        )

        assert len(self.files) > 0, f"Source: Expected > 0 files, got none"

    @property
    def files(self) -> List[Path]:
        files = glob.glob(str(self.source), recursive=True)
        files = [Path(file).resolve() for file in files]
        files = [file for file in files if file.is_file()]

        return files

    @staticmethod
    def get_parser():
        parser = argparse.ArgumentParser()

        inputs = parser.add_argument_group("Inputs")
        outputs = parser.add_argument_group("Outputs")

        inputs.add_argument(
            "--source",
            type=str,
            required=True,
            metavar="PATTERN",
            help="Glob pattern selecting the PCMs to be turned into WAVs",
        )

        outputs.add_argument(
            "--result_dir",
            type=str,
            required=True,
            metavar="DIR",
            help="Path to the directory where converted WAVs will be saved",
        )
        outputs.add_argument(
            "--overwrite",
            action="store_true",
            help="If set, will overwrite on name collision in result directory",
        )
        outputs.add_argument(
            "--verbose",
            action="store_true",
            help="If set, will output progress",
        )

        return parser

    @staticmethod
    def from_args(args):
        source = args.source
        result_dir = args.result_dir
        overwrite = args.overwrite
        verbose = args.verbose

        return PCMToWAVArguments(
            source=source, result_dir=result_dir, overwrite=overwrite, verbose=verbose
        )


def pcm_to_wav(arguments: PCMToWAVArguments):
    inputs = sorted(arguments.files)
    outputs = [
        arguments.result_dir / f"{file.name.split('.')[0]}.wav" for file in inputs
    ]

    arguments.result_dir.mkdir(parents=True, exist_ok=True)

    iterator = zip(inputs, outputs)
    if arguments.verbose:
        iterator = tqdm(
            iterator,
            desc="Converting PCM->WAV",
            total=len(inputs),
            file=sys.stdout,
            ncols=80,
            unit="file",
            unit_scale=True,
        )
    for i, o in iterator:
        if not arguments.overwrite and o.exists():
            continue

        with open(str(i), mode="rb") as f:
            data = f.read()

        with wave.open(str(o), mode="wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(16000)
            f.writeframes(data)


def main():
    parser = PCMToWAVArguments.get_parser()
    args = parser.parse_args()
    arguments = PCMToWAVArguments.from_args(args=args)

    pcm_to_wav(arguments=arguments)


if __name__ == "__main__":
    main()
