import os
import re

from dotenv import load_dotenv

load_dotenv()

_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")

SYSTEM_PROMPT = (
    "You are a Hindi-English transliteration assistant. "
    "Given a transcript that may contain Devanagari script mixed with English:\n"
    "- Convert ALL Devanagari text to colloquial Roman Hindi (as a native speaker would type in WhatsApp/SMS)\n"
    "- Preserve ALL English words exactly as they appear\n"
    "- Preserve punctuation and line breaks\n"
    "- Do NOT translate anything — only romanize\n"
    "- Output only the processed transcript, nothing else"
)


def has_devanagari(text: str) -> bool:
    return bool(_DEVANAGARI_RE.search(text))


def _romanize_anthropic(text: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": text}],
    )
    return message.content[0].text  # type: ignore[index]


def _romanize_azure_openai(text: str) -> str:
    import openai

    client = openai.AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
    )
    deployment = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]
    resp = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    )
    return resp.choices[0].message.content or ""


def _romanize_openai(text: str) -> str:
    import openai

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    )
    return resp.choices[0].message.content or ""


def romanize(text: str) -> str:
    """Romanize Devanagari in text. Returns text unchanged if no Devanagari found."""
    if not has_devanagari(text):
        return text

    attempts = []

    if os.environ.get("ANTHROPIC_API_KEY"):
        attempts.append(_romanize_anthropic)

    azure_chat_vars = ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_CHAT_DEPLOYMENT")
    if all(os.environ.get(v) for v in azure_chat_vars):
        attempts.append(_romanize_azure_openai)

    if os.environ.get("OPENAI_API_KEY"):
        attempts.append(_romanize_openai)

    last_exc: Exception = RuntimeError("No romanization provider configured")
    for fn in attempts:
        try:
            return fn(text)
        except Exception as exc:
            last_exc = exc

    raise last_exc
