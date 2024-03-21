from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import ClassVar, Optional

import azure.cognitiveservices.speech as tts

from .constants import (
    AZURE_OUTPUT_FORMAT_STRING_TO_ENUM,
    AZURE_OUTPUT_FORMAT_STRING_TO_EXTENSION,
)


@dataclass
class AzureSynthesizerConfig:
    DEFAULT_VOICE_NAME: ClassVar[str] = "en-US-JennyNeural"
    DEFAULT_OUTPUT_FORMAT: ClassVar[str] = "raw-16khz-16bit-mono-pcm"

    voice_name: str = DEFAULT_VOICE_NAME
    output_format: str = DEFAULT_OUTPUT_FORMAT

    def __post_init__(self):
        self.voice_name = str(
            self.DEFAULT_VOICE_NAME if self.voice_name is None else self.voice_name
        ).strip()
        self.output_format = (
            str(
                self.DEFAULT_OUTPUT_FORMAT
                if self.output_format is None
                else self.output_format
            )
            .strip()
            .lower()
        )

        assert self.voice_name, f"Voice name: expected a value, got nil"
        assert (
            self.output_format in AZURE_OUTPUT_FORMAT_STRING_TO_ENUM
        ), f"Output format: Unsupported ({self.output_format})"

    @property
    def output_format_enum(self) -> tts.SpeechSynthesisOutputFormat:
        return AZURE_OUTPUT_FORMAT_STRING_TO_ENUM[self.output_format]

    @property
    def extension(self) -> str:
        return AZURE_OUTPUT_FORMAT_STRING_TO_EXTENSION[self.output_format]


@dataclass
class AzureSynthesizerResult:
    DEFAULT_DATA: ClassVar[bytes] = b""
    DEFAULT_MESSAGE: ClassVar[str] = ""

    data: bytes = DEFAULT_DATA
    message: str = DEFAULT_MESSAGE

    def __post_init__(self):
        self.data = bytes(self.DEFAULT_DATA if self.data is None else self.data)
        self.message = str(
            self.DEFAULT_MESSAGE if self.message is None else self.message
        ).strip()

        assert self.message or self.data, f"Expected some data or a message"

    def save(self, path: Path):
        if self.data:
            with open(path, mode="wb+") as f:
                f.write(self.data)
        else:
            print(f"[{path.name}] {self.message}", file=sys.stderr)


class AzureSynthesizer:
    def __init__(
        self, voice_name: Optional[str] = None, output_format: Optional[str] = None
    ):
        self._config = AzureSynthesizerConfig(
            voice_name=voice_name, output_format=output_format
        )

        self._synthesizer = tts.SpeechSynthesizer(
            speech_config=self.speech_config,
        )

    # region Properties
    @property
    def config(self) -> AzureSynthesizerConfig:
        return self._config

    @property
    def voice_name(self) -> str:
        return self.config.voice_name

    @property
    def output_format(self) -> str:
        return self.config.output_format

    @property
    def output_format_enum(self) -> tts.SpeechSynthesisOutputFormat:
        return self.config.output_format_enum

    @property
    def extension(self) -> str:
        return self.config.extension

    @property
    def speech_config(self) -> tts.SpeechConfig:
        speech_config = tts.SpeechConfig(
            subscription=os.environ.get("SPEECH_KEY"),
            region=os.environ.get("SPEECH_REGION"),
        )
        speech_config.speech_synthesis_voice_name = self.voice_name
        speech_config.set_speech_synthesis_output_format(self.output_format_enum)

        return speech_config

    # endregion

    def synthesize(self, text: str) -> AzureSynthesizerResult:
        text = str("" if text is None else text)

        result = self._synthesizer.speak_text(text)
        if result.reason == tts.ResultReason.SynthesizingAudioCompleted:
            return AzureSynthesizerResult(data=result.audio_data)
        elif result.reason == tts.ResultReason.Canceled:
            return AzureSynthesizerResult(
                message=f"Cancelled: {result.cancellation_details.reason.name}"
            )
        else:
            return AzureSynthesizerResult(message=f"Other: {result.reason.name}")
