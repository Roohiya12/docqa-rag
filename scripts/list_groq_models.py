"""List the models your Groq account can use (the catalog changes over time).

Use this to confirm the exact IDs for GROQ_CHAT_MODEL / GROQ_EMBED_MODEL in .env.
"""
import _bootstrap  # noqa: F401
from openai import OpenAI
from docqa import config


def main() -> None:
    if not config.GROQ_API_KEY:
        raise SystemExit("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")
    client = OpenAI(api_key=config.GROQ_API_KEY, base_url=config.GROQ_BASE_URL)
    ids = sorted(m.id for m in client.models.list().data)
    embed = [i for i in ids if "embed" in i.lower()]
    print(f"{len(ids)} models available.\n")
    print("Embedding models:", ", ".join(embed) if embed else "(none listed - check Groq docs)")
    print("\nAll models:")
    for i in ids:
        print("  ", i)
    print(f"\nCurrently configured: chat={config.GROQ_CHAT_MODEL}  embed={config.GROQ_EMBED_MODEL}")


if __name__ == "__main__":
    main()
