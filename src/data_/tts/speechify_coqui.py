from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import ClassVar, Optional

from tqdm import tqdm

from coqui.synthesizer import (
    CoquiSynthesizer,
    CoquiSynthesizerConfig,
)
from utils.speechification import (
    SpeechifyArguments,
    get_distribution_df,
    get_speechify_df,
    get_transcripts_df,
)


@dataclass
class SpeechifyCoquiArguments(SpeechifyArguments):
    DEFAULT_MODEL_NAME: ClassVar[str] = CoquiSynthesizerConfig.DEFAULT_MODEL_NAME
    DEFAULT_SPEAKER: ClassVar[Optional[str]] = CoquiSynthesizerConfig.DEFAULT_SPEAKER
    DEFAULT_TTS_DIR: ClassVar[Path] = CoquiSynthesizerConfig.DEFAULT_TTS_DIR
    DEFAULT_USE_GPU: ClassVar[bool] = False

    model_name: Optional[str] = DEFAULT_MODEL_NAME
    speaker: Optional[str] = DEFAULT_SPEAKER
    tts_dir: Path = DEFAULT_TTS_DIR
    use_gpu: bool = DEFAULT_USE_GPU

    def __post_init__(self):
        self.model_name = str(
            self.DEFAULT_MODEL_NAME if self.model_name is None else self.model_name
        )
        self.speaker = (
            self.DEFAULT_SPEAKER if self.speaker is None else str(self.speaker)
        )
        self.tts_dir = Path(
            self.DEFAULT_TTS_DIR if self.tts_dir is None else self.tts_dir
        )
        self.use_gpu = bool(
            self.DEFAULT_USE_GPU if self.use_gpu is None else self.use_gpu
        )

    @staticmethod
    def get_parser():
        parser = super(SpeechifyCoquiArguments, SpeechifyCoquiArguments).get_parser()

        tts_group = parser.add_argument_group("TTS")

        tts_group.add_argument(
            "--model_name",
            type=str,
            default=SpeechifyCoquiArguments.DEFAULT_MODEL_NAME,
            metavar="STRING",
            help=(
                "Name of the TTS model for synthesizing speech. Default: "
                f"{SpeechifyCoquiArguments.DEFAULT_MODEL_NAME}"
            ),
        )
        tts_group.add_argument(
            "--speaker",
            type=str,
            default=SpeechifyCoquiArguments.DEFAULT_SPEAKER,
            metavar="STRING",
            help="Name of the speaker for synthesizing speech",
        )
        tts_group.add_argument(
            "--tts_dir",
            type=str,
            default=SpeechifyCoquiArguments.DEFAULT_TTS_DIR,
            metavar="DIR",
            help=(
                "Path to the folder where TTS models are saved. Default: "
                f"{SpeechifyCoquiArguments.DEFAULT_TTS_DIR}"
            ),
        )
        tts_group.add_argument(
            "--use_gpu",
            action="store_true",
            help="If set, will use GPU docker image, otherwise the CPU one",
        )

        return parser

    @staticmethod
    def from_args(args) -> "SpeechifyCoquiArguments":
        base_args = super(SpeechifyCoquiArguments, SpeechifyCoquiArguments).from_args(
            args
        )
        model_name = args.model_name
        speaker = args.speaker
        tts_dir = args.tts_dir
        use_gpu = args.use_gpu

        return SpeechifyCoquiArguments(
            **asdict(base_args),
            model_name=model_name,
            speaker=speaker,
            tts_dir=tts_dir,
            use_gpu=use_gpu,
        )


def speechify_coqui(arguments: SpeechifyCoquiArguments):
    if arguments.text:
        synthesizer = CoquiSynthesizer(
            model_name=arguments.model_name,
            speaker=arguments.speaker,
            tts_dir=arguments.tts_dir,
            output_dir=arguments.output_dir,
        )

        output_path = arguments.output_dir / arguments.output_name
        if output_path.exists() and not arguments.overwrite:
            print(
                f"Output path already exists, but overwrite is disabled",
                file=sys.stderr,
            )
        else:
            result = synthesizer.synthesize(
                text=arguments.text,
                path=output_path,
                use_gpu=arguments.use_gpu,
                verbose=arguments.verbose,
            )
            if result and arguments.verbose:
                print(f"{result}", file=sys.stderr)
    else:
        df = get_speechify_df(
            distribution_df=get_distribution_df(
                distribution_path=arguments.distribution_path,
                vendor="coqui",
            ),
            transcripts_df=get_transcripts_df(
                transcripts_path=arguments.transcripts_path
            ),
        )

        iterable = df[["name", "transcript_raw", "tts_name", "tts_engine"]].itertuples(
            index=False
        )
        if arguments.verbose:
            iterable = tqdm(
                iterable,
                desc="Coqui TTS",
                total=df.shape[0],
                file=sys.stdout,
                ncols=80,
                unit="sent",
                unit_scale=True,
            )
        for name, transcript, model_name, speaker in iterable:
            synthesizer = CoquiSynthesizer(
                model_name=model_name,
                speaker=speaker,
                tts_dir=arguments.tts_dir,
                output_dir=arguments.output_dir,
            )

            output_filename = str(name).split(".")[0] + f".wav"
            output_path = synthesizer.output_dir / output_filename
            if output_path.exists() and not arguments.overwrite:
                continue

            result = synthesizer.synthesize(
                text=transcript,
                relative_path=output_filename,
                use_gpu=arguments.use_gpu,
                verbose=arguments.verbose,
            )
            if result:
                print(f"[{output_filename}] {result}", file=sys.stderr)


def main():
    parser = SpeechifyCoquiArguments.get_parser()
    args = parser.parse_args()
    arguments = SpeechifyCoquiArguments.from_args(args=args)

    speechify_coqui()(arguments=arguments)


if __name__ == "__main__":
    main()
