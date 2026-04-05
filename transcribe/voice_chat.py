import os
import tempfile

from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = (
    "You are a helpful voice assistant that understands both English and Hindi. "
    "When the user speaks Hindi (or Roman Hindi), respond in the same language they used. "
    "When they speak English, respond in English. "
    "Keep responses concise and conversational — you are speaking aloud, not writing."
)


# ---------------------------------------------------------------------------
# Chat completion providers
# ---------------------------------------------------------------------------

def _chat_azure(messages: list[dict]) -> str:
    import openai

    client = openai.AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
    )
    resp = client.chat.completions.create(
        model=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
        messages=messages,
    )
    return resp.choices[0].message.content or ""


def _chat_openai(messages: list[dict]) -> str:
    import openai

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
    )
    return resp.choices[0].message.content or ""


def get_response(user_text: str, history: list[dict]) -> str:
    """Send user_text + history to GPT-4o, return assistant reply text."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})

    attempts = []
    azure_vars = ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_CHAT_DEPLOYMENT")
    if all(os.environ.get(v) for v in azure_vars):
        attempts.append(_chat_azure)
    if os.environ.get("OPENAI_API_KEY"):
        attempts.append(_chat_openai)

    last_exc: Exception = RuntimeError("No chat provider configured")
    for fn in attempts:
        try:
            return fn(messages)
        except Exception as exc:
            last_exc = exc
    raise last_exc


# ---------------------------------------------------------------------------
# TTS providers
# ---------------------------------------------------------------------------

def _tts_azure(text: str) -> str:
    import openai

    client = openai.AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
    )
    deployment = os.environ["AZURE_OPENAI_TTS_DEPLOYMENT"]
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    with client.audio.speech.with_streaming_response.create(
        model=deployment,
        voice="nova",
        input=text,
    ) as resp:
        resp.stream_to_file(tmp.name)
    return tmp.name


def _tts_openai(text: str) -> str:
    import openai

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    with client.audio.speech.with_streaming_response.create(
        model="tts-1-hd",
        voice="nova",
        input=text,
    ) as resp:
        resp.stream_to_file(tmp.name)
    return tmp.name


def text_to_speech(text: str) -> str:
    """Convert text to speech. Returns path to mp3 temp file."""
    attempts = []
    if os.environ.get("AZURE_OPENAI_API_KEY") and os.environ.get("AZURE_OPENAI_TTS_DEPLOYMENT"):
        attempts.append(_tts_azure)
    if os.environ.get("OPENAI_API_KEY"):
        attempts.append(_tts_openai)

    last_exc: Exception = RuntimeError("No TTS provider configured")
    for fn in attempts:
        try:
            return fn(text)
        except Exception as exc:
            last_exc = exc
    raise last_exc
