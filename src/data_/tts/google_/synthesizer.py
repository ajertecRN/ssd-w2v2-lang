from dataclasses import dataclass
from pathlib import Path
import sys
from typing import ClassVar, Optional

from google.cloud import texttospeech as tts

from .constants import (
    GOOGLE_AUDIO_ENCODING_STRING_TO_ENUM,
    GOOGLE_AUDIO_ENCODING_STRING_TO_EXTENSION,
)


@dataclass
class GoogleSynthesizerConfig:
    DEFAULT_AUDIO_ENCODING: ClassVar[str] = "linear16"
    DEFAULT_SAMPLE_RATE: ClassVar[int] = 16000
    DEFAULT_VOICE_NAME: ClassVar[str] = "en-US-Standard-A"
    DEFAULT_LANGUAGE_CODE: ClassVar[str] = "en-US"

    audio_encoding: str = DEFAULT_AUDIO_ENCODING
    sample_rate: int = DEFAULT_SAMPLE_RATE
    voice_name: str = DEFAULT_VOICE_NAME

    language_code: ClassVar[str] = DEFAULT_LANGUAGE_CODE
    voice_selection_params: ClassVar[tts.VoiceSelectionParams] = None
    audio_config: ClassVar[tts.AudioConfig] = None

    def __post_init__(self):
        self.audio_encoding = (
            str(
                self.DEFAULT_AUDIO_ENCODING
                if self.audio_encoding is None
                else self.audio_encoding
            )
            .strip()
            .lower()
        )
        self.sample_rate = int(
            self.DEFAULT_SAMPLE_RATE if self.sample_rate is None else self.sample_rate
        )
        self.voice_name = str(
            self.DEFAULT_VOICE_NAME if self.voice_name is None else self.voice_name
        ).strip()

        assert (
            self.audio_encoding in GOOGLE_AUDIO_ENCODING_STRING_TO_ENUM
        ), f"Audio encoding: Unsupported ({self.audio_encoding})"
        assert (
            self.sample_rate > 0
        ), f"Sample rate: Expected > 0, got {self.sample_rate}"
        assert self.voice_name, f"Voice name: Expected a value, got nil"

        self.language_code = "-".join(self.voice_name.split("-")[:2])
        self.voice_selection_params = tts.VoiceSelectionParams(
            language_code=self.language_code, name=self.voice_name
        )
        self.audio_config = tts.AudioConfig(
            audio_encoding=GOOGLE_AUDIO_ENCODING_STRING_TO_ENUM[self.audio_encoding],
            sample_rate_hertz=self.sample_rate,
        )

    @property
    def extension(self):
        return GOOGLE_AUDIO_ENCODING_STRING_TO_EXTENSION[self.audio_encoding]


@dataclass
class GoogleSynthesizerResult:
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


class GoogleSynthesizer:
    def __init__(
        self,
        audio_encoding: Optional[str] = None,
        sample_rate: Optional[int] = None,
        voice_name: Optional[str] = None,
    ):
        self._config = GoogleSynthesizerConfig(
            audio_encoding=audio_encoding,
            sample_rate=sample_rate,
            voice_name=voice_name,
        )
        self._client = tts.TextToSpeechClient()

    # region Properties
    @property
    def config(self) -> GoogleSynthesizerConfig:
        return self._config

    @property
    def audio_encoding(self) -> str:
        return self.config.audio_encoding

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    @property
    def voice_name(self) -> str:
        return self.config.voice_name

    @property
    def language_code(self) -> str:
        return self.config.language_code

    @property
    def voice_selection_params(self) -> tts.VoiceSelectionParams:
        return self.config.voice_selection_params

    @property
    def audio_config(self) -> tts.AudioConfig:
        return self.config.audio_config

    @property
    def extension(self) -> str:
        return self.config.extension

    # endregion

    def synthesize(self, text: str) -> GoogleSynthesizerResult:
        text = str(text or "")
        synthesis_input = tts.SynthesisInput(text=text)

        try:
            response = self._client.synthesize_speech(
                request={
                    "input": synthesis_input,
                    "voice": self.voice_selection_params,
                    "audio_config": self.audio_config,
                }
            )
        except Exception as error:
            return GoogleSynthesizerResult(message=f"Response: {error}".strip())

        if not response.audio_content:
            return GoogleSynthesizerResult(message="Empty response")

        return GoogleSynthesizerResult(data=response.audio_content)
