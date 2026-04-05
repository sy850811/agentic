from .romanizer import romanize
from .whisper_client import WhisperClient


class TranscriptionPipeline:
    def __init__(self) -> None:
        self._whisper = WhisperClient()

    def run(self, audio_path: str) -> str:
        raw = self._whisper.transcribe(audio_path)
        return romanize(raw)
