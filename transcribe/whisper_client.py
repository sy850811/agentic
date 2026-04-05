import os
from typing import Protocol

from dotenv import load_dotenv

from .chunker import chunk_audio, cleanup_chunks, needs_chunking

load_dotenv()

# Seeding with Roman Hindi examples nudges Whisper to output Roman script
# instead of Devanagari for Hindi words.
INITIAL_PROMPT = (
    "Yaar, aaj meeting mein kya hua? Bohot interesting tha. "
    "Main theek hoon, aap kaise hain? Haan, bilkul sahi keh rahe ho. "
    "Theek hai, let's move on. Dekho, problem yeh hai."
)


class _WhisperProvider(Protocol):
    def transcribe_file(self, audio_path: str) -> str: ...


class _OpenAIProvider:
    def __init__(self) -> None:
        import openai

        self._client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def transcribe_file(self, audio_path: str) -> str:
        with open(audio_path, "rb") as f:
            result = self._client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="text",
                prompt=INITIAL_PROMPT,
            )
        return result  # type: ignore[return-value]


class _AzureProvider:
    def __init__(self) -> None:
        import openai

        self._client = openai.AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        )
        self._deployment = os.environ["AZURE_OPENAI_WHISPER_DEPLOYMENT"]

    def transcribe_file(self, audio_path: str) -> str:
        with open(audio_path, "rb") as f:
            result = self._client.audio.transcriptions.create(
                model=self._deployment,
                file=f,
                response_format="text",
                prompt=INITIAL_PROMPT,
            )
        return result  # type: ignore[return-value]


def _build_providers() -> list[_WhisperProvider]:
    providers: list[_WhisperProvider] = []

    azure_vars = ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_WHISPER_DEPLOYMENT")
    if all(os.environ.get(v) for v in azure_vars):
        try:
            providers.append(_AzureProvider())
        except Exception:
            pass

    if os.environ.get("OPENAI_API_KEY"):
        try:
            providers.append(_OpenAIProvider())
        except Exception:
            pass

    return providers


class WhisperClient:
    def __init__(self) -> None:
        self._providers = _build_providers()
        if not self._providers:
            raise RuntimeError(
                "No Whisper provider configured. "
                "Set OPENAI_API_KEY or Azure OpenAI env vars in .env"
            )

    def transcribe(self, audio_path: str) -> str:
        if needs_chunking(audio_path):
            return self._transcribe_chunked(audio_path)
        return self._transcribe_single(audio_path)

    def _transcribe_single(self, audio_path: str) -> str:
        last_exc: Exception = RuntimeError("No providers available")
        for provider in self._providers:
            try:
                return provider.transcribe_file(audio_path)
            except Exception as exc:
                last_exc = exc
        raise last_exc

    def _transcribe_chunked(self, audio_path: str) -> str:
        chunks = chunk_audio(audio_path)
        try:
            parts = [self._transcribe_single(c) for c in chunks]
            return " ".join(p.strip() for p in parts if p.strip())
        finally:
            cleanup_chunks(chunks)
