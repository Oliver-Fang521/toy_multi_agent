"""Read a service token from hidden stdin; select only needed existing local env keys."""
import json
import sys
from pathlib import Path

import modal


def main():
    payload = json.loads(sys.stdin.readline())
    source = Path(payload["env_path"])
    allowed = {"OPENAI_API_KEY", "OPENAI_BASE_URL", "SERPER_API_KEY", "SERPAPI_API_KEY"}
    values = {}
    for line in source.read_text().splitlines():
        key, separator, value = line.partition("=")
        key = key.removeprefix("export ").strip()
        if separator and key in allowed:
            value = value.strip().strip('"\'')
            if value:
                values[key] = value
    if "OPENAI_API_KEY" not in values:
        raise RuntimeError("OpenAI credential is missing")
    values["AGENT_SERVICE_TOKEN"] = payload["service_token"]
    modal.Secret.objects.create("toy-agent-web-runtime", values)
    print("Runtime secret configured; no credential values printed.")


if __name__ == "__main__":
    main()
