"""Quick test: verify both API keys work before running the pipeline."""

import os
from dotenv import load_dotenv

load_dotenv()


print("\nTesting Anthropic...")
try:
    from anthropic import Anthropic
    ant = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    resp = ant.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=10,
        messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
    )
    print(f"  [OK] Anthropic works: {resp.content[0].text}")
except Exception as e:
    print(f"  [FAIL] Anthropic failed: {e}")

print("\nDone.")