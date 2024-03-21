import argparse
from dataclasses import dataclass
import glob
from pathlib import Path
import shutil

from utils import parse_arctic_prompts


@dataclass
class AssembleCMUArguments:
    cmu_dir: Path
    result_dir: Path

    def __post_init__(self):
        self.cmu_dir = Path(self.cmu_dir).resolve()
        self.result_dir = Path(self.result_dir).resolve()

        assert self.cmu_dir.exists(), f"CMU directory: Expected it to exist"

    @staticmethod
    def get_parser():
        parser = argparse.ArgumentParser()

        inputs = parser.add_argument_group("Inputs")
        outputs = parser.add_argument_group("Outputs")

        inputs.add_argument(
            "--cmu_dir",
            type=str,
            required=True,
            metavar="DIR",
            help="Path to the CMU speaker directory (ex. cmu_us_aew_arctic)",
        )

        outputs.add_argument(
            "--result_dir",
            type=str,
            required=True,
            metavar="DIR",
            help="Path to the directory where results will be saved",
        )

        return parser

    @staticmethod
    def from_args(args) -> "AssembleCMUArguments":
        cmu_dir = args.cmu_dir
        result_dir = args.result_dir

        return AssembleCMUArguments(cmu_dir=cmu_dir, result_dir=result_dir)


def assemble_cmu(arguments: AssembleCMUArguments):
    data_dir = arguments.result_dir / "data"
    audio_dir = arguments.result_dir / "audio" / "original"

    data_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    prompts_file = arguments.cmu_dir / "etc" / "txt.done.data"
    if prompts_file.exists():
        with open(prompts_file, mode="r", encoding="utf8", errors="replace") as f:
            prompts = f.read()

        transcripts = parse_arctic_prompts(prompts=prompts)
        transcripts.to_csv(data_dir / "transcripts.csv", index=False)

    utterance_dir = arguments.cmu_dir / "wav"
    utterance_files = glob.glob(f"{utterance_dir}/*.wav")
    utterance_files = [Path(utterance) for utterance in utterance_files]
    utterance_files = [
        utterance for utterance in utterance_files if utterance.is_file()
    ]
    utterance_files = sorted(utterance_files)
    for utterance in utterance_files:
        shutil.copyfile(utterance, audio_dir / utterance.name)


def main():
    parser = AssembleCMUArguments.get_parser()
    args = parser.parse_args()
    arguments = AssembleCMUArguments.from_args(args=args)

    assemble_cmu(arguments=arguments)


if __name__ == "__main__":
    main()
