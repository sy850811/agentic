"""
Open Brain connector.

Captures thoughts to Supabase via the upsert_thought RPC.
Open Brain handles embeddings server-side — we just send the text.

Required env vars:
    SUPABASE_URL         — Project URL from Supabase dashboard
    SUPABASE_SECRET_KEY  — Secret key (not the publishable/anon key)
"""

import os

from dotenv import load_dotenv

load_dotenv()


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
    """Save a thought to Open Brain. Raises on failure."""
    if not content or not content.strip():
        raise ValueError("Cannot capture an empty thought")

    client = _get_supabase_client()

    result = client.rpc(
        "upsert_thought",
        {"p_content": content, "p_payload": {"metadata": metadata or {}}},
    ).execute()

    if not (result.data and result.data.get("id")):
        raise RuntimeError("Open Brain: upsert returned no row id")
