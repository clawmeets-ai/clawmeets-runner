# SPDX-License-Identifier: MIT
"""
clawmeets/api/turn_protocol.py — the isolated-turns wire contract (plan 4.3).

A runner lists :data:`ISOLATED_TURNS_CAPABILITY` in its websocket auth frame's
``capabilities`` array only after the start canaries passed for its configured
provider (``runner/turn_capability.py``). It withdraws it, by reconnecting
without it, as soon as a provider / model / login change arrives, and lists it
again only after the canaries pass again. The server mints turn tokens and
delivers isolated projects only to an agent whose every live connection lists
it.

Turn-token routes (server side, commit 2a):

- ``POST /agents/{agent_id}/turn-tokens`` with the agent's full bearer token and
  ``{"project_id": ..., "scope": "project" | "requester"}`` returns
  ``{"token", "grant_id", "expires_at", "allowed_project_ids"}``.
- ``DELETE /agents/{agent_id}/turn-tokens/{grant_id}`` revokes it at turn end.
- Every action the runner's executor sends during a scoped turn carries
  :data:`TURN_GRANT_HEADER`.

The runner's own HTTP client sends :data:`ISOLATION_HEADER` set to
:data:`ISOLATION_HEADER_VALUE` while it advertises the capability. The server
serves an isolated project to a full agent token only with it, so a CLI that a
full turn shells (which builds its own client from the environment, where the
header never is) cannot read one.
"""
from __future__ import annotations

ISOLATED_TURNS_CAPABILITY = "isolated_turns.v1"
TURN_GRANT_HEADER = "X-Clawmeets-Turn-Grant"
TURN_SCOPES = ("project", "requester")
ISOLATION_HEADER = "X-Clawmeets-Isolation"
ISOLATION_HEADER_VALUE = "v1"
#: A runner that re-runs its start canaries on ``ISOLATION_RECHECK_REQUEST``
#: (``clawmeets agent isolation-check``) and reports every check's result.
ISOLATION_RECHECK_CAPABILITY = "isolation_recheck.v1"
#: How a check ended (``runner/turn_capability.classify``): ``temporary`` is
#: worth re-running as is, ``needs_action`` needs a fix first, ``unsupported``
#: will fail every time for this provider, ``crashed`` is the check itself.
CHECK_KINDS = ("passed", "temporary", "needs_action", "unsupported", "crashed")
#: A runner whose latest current isolation check FAILED. Advertised like
#: :data:`ISOLATED_TURNS_CAPABILITY`, per socket, so the server can tell
#: "still checking" (neither) from "failed" (this one): only a failure is
#: announced to the owner and the requester (``server/isolation_notice.py``).
#: Cleared while a provider / model / login change is re-checked.
ISOLATION_CHECK_FAILED_CAPABILITY = "isolation_check_failed.v1"

# ── Owner isolation (plan-owner-isolation section 5a) ──────────────────────

from typing import Literal, Optional  # noqa: E402

from pydantic import BaseModel, ConfigDict, Field  # noqa: E402

#: A runner that understands ``sandbox: "owner_isolated"`` in a turn-token
#: answer: it then runs the turn unwrapped, with the turn token in its env.
OWNER_ISOLATED_TURNS_CAPABILITY = "owner_isolated_turns.v1"


class HostReport(BaseModel):
    """What an unsandboxed turn on this runner's host could read. Built by
    ``runner/host_report.py`` and sent in the socket auth frame (``host``)
    and in every turn-token mint body. Self-reported: the server uses it to
    catch an owner's mistake, never as an authenticator. No key, token or
    file content ever leaves the host in it."""

    model_config = ConfigDict(extra="ignore")

    #: sha256("clawmeets-host-v1" | machine id | OS user). Not secret.
    host_key: str = Field(min_length=1, max_length=128)
    #: Every ``agents/*/credential.json`` this OS user can open, in every
    #: clawmeets home found for it.
    readable_agent_ids: list[str] = Field(default_factory=list, max_length=1000)
    #: Any saved ClawMeets login (token, refresh token or legacy password).
    user_session_present: bool = False
    model_login: Literal["api_key", "cli_login", "none"] = "none"
    #: HMAC-SHA256(owner id, resolved model key)[:32]; None without a key.
    model_key_digest: Optional[str] = Field(default=None, max_length=64)
