"""Infrastructure-only JSON on hidden stdin. Never reads env files.

Invoking main transmits infrastructure secrets to Modal; separate authorization is required.
"""

import base64
import json
import sys

from gaia_agent.interactive.credential_crypto import KeyCipher
from gaia_agent.interactive.credential_runtime import InteractiveRuntimeSettings
from gaia_agent.interactive.credential_vault import CredentialError

INFRASTRUCTURE_KEYS = frozenset(
    {
        "AGENT_SERVICE_TOKEN",
        "AGENT_CREDENTIAL_KEYRING",
        "AGENT_CREDENTIAL_ACTIVE_VERSION",
        "AGENT_MODEL_BASE_URL",
    }
)


def infrastructure_values(payload):
    try:
        if (
            not isinstance(payload, dict)
            or set(payload) != INFRASTRUCTURE_KEYS
            or not all(isinstance(v, str) for v in payload.values())
        ):
            raise ValueError("configuration")
        token = payload["AGENT_SERVICE_TOKEN"]
        if len(token) < 16 or any(ord(c) < 33 or ord(c) > 126 for c in token):
            raise ValueError("configuration")
        raw = json.loads(payload["AGENT_CREDENTIAL_KEYRING"])
        if not isinstance(raw, dict):
            raise TypeError("configuration")
        KeyCipher(
            {version: base64.b64decode(value, validate=True) for version, value in raw.items()},
            payload["AGENT_CREDENTIAL_ACTIVE_VERSION"],
        )
        InteractiveRuntimeSettings(base_url=payload["AGENT_MODEL_BASE_URL"])
    except (ValueError, TypeError, CredentialError):
        raise CredentialError("credential_configuration", 503) from None
    return {name: payload[name] for name in INFRASTRUCTURE_KEYS}


def main():
    import modal

    try:
        values = infrastructure_values(json.loads(sys.stdin.readline(65537)))
    except (ValueError, CredentialError):
        raise SystemExit("credential_configuration") from None
    modal.Secret.objects.create("toy-agent-web-byok-runtime", values)
    print("Infrastructure configured. No model/search credentials copied.")


if __name__ == "__main__":
    main()
