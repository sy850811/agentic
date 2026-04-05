import os

from dotenv import load_dotenv

load_dotenv()


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
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
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
