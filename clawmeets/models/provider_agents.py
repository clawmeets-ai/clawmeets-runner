# SPDX-License-Identifier: MIT
"""
clawmeets/models/provider_agents.py

One fixed-name agent per way this computer can reach a model provider: each
signed-in model CLI, and each provider the user gave an API key for. The user
meets them as "model agents": each speaks for its model and for what that model
has seen of the user's work. :data:`MODEL_AGENT_DEFINITION` is the one sentence
that says so, reused wherever a user reads the term (the starter to-dos in
``desk_todo`` and the assistant's start-view samples).

Names are fixed so text written before install (the seed onboarding to-do in
``desk_todo``) can name them without knowing what a given machine has, and so
re-running the installer finds the same agents instead of adding a second
``codex``.

CLI entries are derived from :data:`clawmeets.doctor.MODEL_CLIS`, so adding a
CLI to doctor adds an agent here, in doctor's order. Key entries cover
:data:`model_config.KEYED_PROVIDERS` minus ``openrouter-native``, which takes
the same key as ``openrouter-api`` and would be a duplicate.
``tests/test_provider_agents.py`` pins both.

Registered by ``clawmeets agent provider register`` (the installer's
"Connecting your model providers" step). Lives under ``models/`` so the runner
wheel ships it through the subtree copy and the server can import it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from clawmeets.doctor import MODEL_CLIS


@dataclass(frozen=True)
class ProviderAgent:
    # Short name; the server prefixes "{username}-". No hyphens: the server
    # rejects them, because the hyphen separates the username from the name.
    name: str
    # The ``local_settings.llm_provider`` the agent registers with.
    provider: str
    kind: Literal["cli", "key"]
    label: str
    # Key entries only: env vars whose value the key prompt offers to reuse.
    key_env: tuple[str, ...] = ()


_KEY_AGENTS: tuple[ProviderAgent, ...] = (
    ProviderAgent("claude_api", "claude-api", "key", "Anthropic API", ("ANTHROPIC_API_KEY",)),
    ProviderAgent("openai_api", "openai-api", "key", "OpenAI API", ("OPENAI_API_KEY",)),
    ProviderAgent(
        "gemini_api", "gemini-api", "key", "Gemini API", ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    ),
    ProviderAgent("openrouter", "openrouter-api", "key", "OpenRouter", ("OPENROUTER_API_KEY",)),
)

PROVIDER_AGENTS: tuple[ProviderAgent, ...] = (
    *(ProviderAgent(spec.id, spec.provider, "cli", spec.label) for spec in MODEL_CLIS),
    *_KEY_AGENTS,
)

PROVIDER_TEAM = "Model agents"
# Earlier names of PROVIDER_TEAM. ``agent provider register`` swaps any of these
# on an existing agent for the current one, so a re-run of the installer moves
# the sidebar group to the new name instead of splitting it in two.
LEGACY_PROVIDER_TEAMS: tuple[str, ...] = ("Model providers",)

# The one definition of "model agent", word for word wherever a user reads the
# term. Written in the owner's voice because it is quoted inside prompts the
# owner sends, including the first of the assistant's start-view samples
# (``web/frontend/.../assistantStarterSamples.ts``, pinned to this text by
# ``tests/test_provider_agents.py``). The last sentence answers "my assistant runs on Claude, so how is
# it different from `claude`?", which is also why a panel of model agents is
# model-diverse while a panel of role agents may all run on one model.
MODEL_AGENT_DEFINITION = (
    "Model agents are the agents named after the AI models I connected at "
    "install (claude, codex, gemini, ..., plus one per API key). Each one has no "
    "job of its own: it speaks for its model and for what that model has "
    "already seen of my work. My other agents, like my assistant and "
    "thought_partner, each have a job, and each one runs on one of those models."
)

def cli_agents() -> tuple[ProviderAgent, ...]:
    return tuple(a for a in PROVIDER_AGENTS if a.kind == "cli")


def key_agents() -> tuple[ProviderAgent, ...]:
    return tuple(a for a in PROVIDER_AGENTS if a.kind == "key")


def provider_agent_names() -> tuple[str, ...]:
    return tuple(a.name for a in PROVIDER_AGENTS)


def description(agent: ProviderAgent) -> str:
    """The one-line description the agent registers with. The owner can edit
    it later like any agent's."""
    return (
        f"Model agent for {agent.label}: speaks for this model and what it has "
        "seen of your work."
    )
