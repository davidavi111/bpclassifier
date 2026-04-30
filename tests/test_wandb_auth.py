"""W&B authentication smoke test — opt-in (skipped by default).

Run with:  pytest -m smoke
"""

import json
import os
import urllib.request

import pytest


@pytest.mark.smoke
def test_wandb_key_works_against_api():
    """Actually queries W&B's GraphQL endpoint with the configured key."""
    if not os.environ.get("WANDB_API_KEY"):
        pytest.skip("WANDB_API_KEY not set — skipping live W&B check.")

    req = urllib.request.Request(
        "https://api.wandb.ai/graphql",
        data=json.dumps({"query": "{ viewer { username } }"}).encode(),
        headers={
            "Authorization": f"Bearer {os.environ['WANDB_API_KEY']}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310 - hardcoded URL is fine
        result = json.loads(resp.read().decode())

    assert "data" in result, f"Unexpected W&B response: {result}"
    assert "viewer" in result["data"], f"No viewer field in response: {result}"
    username = result["data"]["viewer"]["username"]
    assert isinstance(username, str) and len(username) > 0
    print(f"\nW&B authenticated as: {username}")  # username is not a secret
