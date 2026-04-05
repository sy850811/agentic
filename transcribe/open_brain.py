"""
Open Brain connector.

Captures thoughts to the user's Open Brain (Supabase + pgvector).
Uses the same OpenAI/Azure credentials already configured for Whisper
to generate 1536-dim embeddings, then calls the upsert_thought RPC
for deduplication before inserting.

Required env vars (add to .env):
    SUPABASE_URL         — Project URL from Supabase dashboard
    SUPABASE_SECRET_KEY  — Secret key (not the anon/publishable key)
"""

import os

from dotenv import load_dotenv

load_dotenv()

_EMBEDDING_MODEL = "text-embedding-3-small"  # 1536 dims, matches schema


# ---------------------------------------------------------------------------
# Embedding generation — reuses existing OpenAI / Azure credentials
# ---------------------------------------------------------------------------

def _embed_azure(text: str) -> list[float]:
    import openai

    client = openai.AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
    )
    deployment = os.environ.get("AZURE_OPENAI_EMBED_DEPLOYMENT", _EMBEDDING_MODEL)
    resp = client.embeddings.create(model=deployment, input=text)
    return resp.data[0].embedding


def _embed_openai(text: str) -> list[float]:
    import openai

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.embeddings.create(model=_EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding


def _generate_embedding(text: str) -> list[float]:
    azure_embed_vars = ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT")
    if all(os.environ.get(v) for v in azure_embed_vars):
        try:
            return _embed_azure(text)
        except Exception as exc:
            raise RuntimeError(f"Azure embedding failed: {exc}") from exc

    if os.environ.get("OPENAI_API_KEY"):
        return _embed_openai(text)

    raise RuntimeError(
        "No embedding provider configured. "
        "Set OPENAI_API_KEY or Azure OpenAI vars in .env"
    )


# ---------------------------------------------------------------------------
# Supabase write
# ---------------------------------------------------------------------------

def _get_supabase_client():
    from supabase import create_client

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SECRET_KEY")

    if not url or not key:
        raise RuntimeError(
            "Open Brain not configured. "
            "Add SUPABASE_URL and SUPABASE_SECRET_KEY to .env"
        )
    return create_client(url, key)


def capture(content: str, metadata: dict | None = None) -> None:
    """
    Save a thought to Open Brain. Raises on failure — caller decides what to show.

    Calls the upsert_thought RPC (dedup by content fingerprint),
    then patches in the embedding for semantic search.
    """
    if not content or not content.strip():
        raise ValueError("Cannot capture an empty thought")

    client = _get_supabase_client()
    payload = {"metadata": metadata or {}}

    # Step 1: upsert (dedup by content fingerprint)
    result = client.rpc(
        "upsert_thought",
        {"p_content": content, "p_payload": payload},
    ).execute()

    row_id = result.data.get("id") if result.data else None
    if not row_id:
        raise RuntimeError("Open Brain: upsert returned no row id")

    # Step 2: generate embedding and patch the row
    embedding = _generate_embedding(content)
    client.table("thoughts").update({"embedding": embedding}).eq("id", row_id).execute()
