from dataclasses import asdict, dataclass
import sys
from typing import ClassVar

from tqdm import tqdm

from amazon.synthesizer import (
    AMAZON_ENGINES,
    AMAZON_OUTPUT_FORMATS,
    AmazonSynthesizer,
    AmazonSynthesizerConfig,
)
from utils.speechification import (
    SpeechifyArguments,
    get_distribution_df,
    get_speechify_df,
    get_transcripts_df,
)


@dataclass
class SpeechifyAmazonArguments(SpeechifyArguments):
    DEFAULT_ENGINE: ClassVar[str] = AmazonSynthesizerConfig.DEFAULT_ENGINE
    DEFAULT_OUTPUT_FORMAT: ClassVar[str] = AmazonSynthesizerConfig.DEFAULT_OUTPUT_FORMAT
    DEFAULT_SAMPLE_RATE: ClassVar[int] = AmazonSynthesizerConfig.DEFAULT_SAMPLE_RATE
    DEFAULT_VOICE_NAME: ClassVar[str] = AmazonSynthesizerConfig.DEFAULT_VOICE_NAME
    DEFAULT_PROFILE_NAME: ClassVar[str] = AmazonSynthesizerConfig.DEFAULT_PROFILE_NAME

    engine: str = DEFAULT_ENGINE
    output_format: str = DEFAULT_OUTPUT_FORMAT
    sample_rate: int = DEFAULT_SAMPLE_RATE
    voice_name: str = DEFAULT_VOICE_NAME
    profile_name: str = DEFAULT_PROFILE_NAME

    def __post_init__(self):
        self.engine = str(self.DEFAULT_ENGINE if self.engine is None else self.engine)
        self.output_format = str(
            self.DEFAULT_OUTPUT_FORMAT
            if self.output_format is None
            else self.output_format
        )
        self.sample_rate = int(
            self.DEFAULT_SAMPLE_RATE if self.sample_rate is None else self.sample_rate
        )
        self.voice_name = str(
            self.DEFAULT_VOICE_NAME if self.voice_name is None else self.voice_name
        )
        self.profile_name = str(
            self.DEFAULT_PROFILE_NAME
            if self.profile_name is None
            else self.profile_name
        )

    @staticmethod
    def get_parser():
        parser = super(SpeechifyAmazonArguments, SpeechifyAmazonArguments).get_parser()

        tts_group = parser.add_argument_group("TTS")

        tts_group.add_argument(
            "--engine",
            type=str,
            choices=AMAZON_ENGINES,
            default=SpeechifyAmazonArguments.DEFAULT_ENGINE,
            metavar="STRING",
            help=(
                "Name of the voice engine. Default: "
                f"{SpeechifyAmazonArguments.DEFAULT_ENGINE }"
            ),
        )
        tts_group.add_argument(
            "--output_format",
            type=str,
            choices=AMAZON_OUTPUT_FORMATS,
            default=SpeechifyAmazonArguments.DEFAULT_OUTPUT_FORMAT,
            metavar="STRING",
            help=(
                "Audio format of the synthesized speech. Default: "
                f"{SpeechifyAmazonArguments.DEFAULT_OUTPUT_FORMAT}"
            ),
        )
        tts_group.add_argument(
            "--sample_rate",
            type=int,
            default=SpeechifyAmazonArguments.DEFAULT_SAMPLE_RATE,
            metavar="HERTZ",
            help=(
                "Sample rate of the synthesized speech. Default: "
                f"{SpeechifyAmazonArguments.DEFAULT_SAMPLE_RATE}"
            ),
        )
        tts_group.add_argument(
            "--voice_name",
            type=str,
            default=SpeechifyAmazonArguments.DEFAULT_VOICE_NAME,
            metavar="STRING",
            help=(
                "Name of the TTS voice for synthesizing speech. Default: "
                f"{SpeechifyAmazonArguments.DEFAULT_VOICE_NAME}"
            ),
        )
        tts_group.add_argument(
            "--profile_name",
            type=str,
            default=SpeechifyAmazonArguments.DEFAULT_PROFILE_NAME,
            help=(
                "Name of the profile used for authentication. Default: "
                f"{SpeechifyAmazonArguments.DEFAULT_PROFILE_NAME}"
            ),
        )

        return parser

    @staticmethod
    def from_args(args) -> "SpeechifyAmazonArguments":
        base_args = super(SpeechifyAmazonArguments, SpeechifyAmazonArguments).from_args(
            args
        )
        engine = args.engine
        output_format = args.output_format
        sample_rate = args.sample_rate
        voice_name = args.voice_name
        profile_name = args.profile_name

        return SpeechifyAmazonArguments(
            **asdict(base_args),
            engine=engine,
            output_format=output_format,
            sample_rate=sample_rate,
            voice_name=voice_name,
            profile_name=profile_name,
        )


def speechify_amazon(arguments: SpeechifyAmazonArguments):
    if arguments.text:
        synthesizer = AmazonSynthesizer(
            engine=arguments.engine,
            output_format=arguments.output_format,
            sample_rate=arguments.sample_rate,
            voice_name=arguments.voice_name,
            profile_name=arguments.profile_name,
        )

        output_path = arguments.output_dir / arguments.output_name
        if output_path.exists() and not arguments.overwrite:
            print(
                f"Output path already exists, but overwrite is disabled",
                file=sys.stderr,
            )
        else:
            result = synthesizer.synthesize(text=arguments.text)
            result.save(output_path)
    else:
        df = get_speechify_df(
            distribution_df=get_distribution_df(
                distribution_path=arguments.distribution_path, vendor="amazon"
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
                desc="Amazon TTS",
                total=df.shape[0],
                file=sys.stdout,
                ncols=80,
                unit="sent",
                unit_scale=True,
            )
        for name, transcript, voice_name, engine in iterable:
            synthesizer = AmazonSynthesizer(
                engine=engine,
                output_format=arguments.output_format,
                sample_rate=arguments.sample_rate,
                voice_name=voice_name,
                profile_name=arguments.profile_name,
            )
            output_filename = str(name).split(".")[0] + f".{synthesizer.extension}"
            output_path = arguments.output_dir / output_filename

            if output_path.exists() and not arguments.overwrite:
                continue

            result = synthesizer.synthesize(text=transcript)
            result.save(output_path)


def main():
    parser = SpeechifyAmazonArguments.get_parser()
    args = parser.parse_args()
    arguments = SpeechifyAmazonArguments.from_args(args=args)

    speechify_amazon(arguments=arguments)


if __name__ == "__main__":
    main()
