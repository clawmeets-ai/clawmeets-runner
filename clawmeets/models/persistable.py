# SPDX-License-Identifier: MIT
"""
clawmeets/models/persistable.py
Base class with Active Record persistence for Agent and Assistant.

## Local-Only Mutations (Intentionally Not Distributed)

Unlike Project/Chatroom (which mutate via the distributed changelog),
Agent/Assistant mutations write directly to card.json and are LOCAL-ONLY.
These changes:
- Do NOT flow through the changelog
- Are NOT visible to other agents
- Include: status updates, heartbeats, registration

This is intentional - participant metadata is runner-local state
(e.g., "am I online?"), while project/chatroom state is shared
across all participants and must be coordinated via the changelog.

## Provided Methods

- Class methods: get, get_by_name, list_all, register, verify_token
- Instance methods: save, update_status, heartbeat, to_response

All state is read from filesystem (card.json) on each property access.
"""
from __future__ import annotations

import hashlib
import secrets
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Optional

from ..api.responses import AgentResponse, AgentStatus
from ..utils.file_io import FileUtil
from ..utils.validation import validate_name
from .participant import Participant, ParticipantRole

if TYPE_CHECKING:
    from typing_extensions import Self

    from .context import ModelContext


class PersistableParticipant(Participant, ABC):
    """Base class with Active Record persistence for Agent/Assistant.

    Subclasses must set:
        _role_subdir: ClassVar[str]  # "agents" or "assistants"

    All state is read from the filesystem (card.json) on each property access.
    This ensures the model always reflects the current state on disk.
    """

    # Subclasses must override this
    _role_subdir: ClassVar[str]

    def __init__(
        self,
        id: str,
        model_ctx: "ModelContext",
    ) -> None:
        """Initialize a PersistableParticipant.

        Args:
            id: The participant's unique identifier
            model_ctx: ModelContext for filesystem access
        """
        super().__init__()
        self._id = id
        self._model_ctx = model_ctx

    # ─────────────────────────────────────────────────────────────────────────
    # Filesystem Helpers
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def _get_dir(
        cls,
        ctx: "ModelContext",
        name: str,
        participant_id: str,
    ) -> Path:
        """Return participant directory path.

        Directory creation is handled by FileUtil.write() with ensure_dir=True.
        Uses cls._role_subdir directly to avoid role key conversion.

        Args:
            ctx: ModelContext for filesystem access
            name: Participant name
            participant_id: Participant ID

        Returns:
            Path to participant directory
        """
        return ctx.participants_dir / cls._role_subdir / f"{name}-{participant_id}"

    @classmethod
    def _list_dirs(cls, ctx: "ModelContext") -> list[Path]:
        """List all participant directories for this role.

        Uses cls._role_subdir directly (e.g., "agents", "assistants").

        Args:
            ctx: ModelContext for filesystem access

        Returns:
            Sorted list of directory paths with card.json
        """
        role_dir = ctx.participants_dir / cls._role_subdir
        if not role_dir.exists():
            return []
        return sorted(
            d for d in role_dir.iterdir()
            if d.is_dir() and (d / "card.json").exists()
        )

    @property
    def card_path(self) -> Optional[Path]:
        """Get path to this participant's card.json.

        Uses the known _role_subdir to find the directory without scanning all roles.
        """
        search_dir = self._model_ctx.participants_dir / self._role_subdir
        if not search_dir.exists():
            return None
        matches = list(search_dir.glob(f"*-{self._id}"))
        return matches[0] / "card.json" if matches else None

    def _load_card(self) -> dict:
        """Load card.json from filesystem.

        Returns:
            Dict of card data, or empty dict if not found
        """
        path = self.card_path
        if path is None:
            return {}
        card = FileUtil.read(path, "json")
        return card if card else {}

    def _save_card(self, data: dict) -> None:
        """Save card.json to filesystem.

        Args:
            data: Card data to save
        """
        part_dir = self._get_dir(
            self._model_ctx,
            data.get("name", "unknown"),
            self._id,
        )
        FileUtil.write(part_dir / "card.json", data, "json", atomic=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Abstract Properties (from Participant)
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        """Get name from filesystem."""
        return self._load_card().get("name", "")

    @property
    @abstractmethod
    def role(self) -> ParticipantRole:
        """Return the participant role."""
        pass

    # ─────────────────────────────────────────────────────────────────────────
    # Common Properties
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def description(self) -> str:
        """Get description from filesystem."""
        return self._load_card().get("description", "")

    @property
    def capabilities(self) -> list[str]:
        """Get capabilities from filesystem."""
        return self._load_card().get("capabilities", [])

    @property
    def status(self) -> AgentStatus:
        """Get status from filesystem."""
        status_str = self._load_card().get("status", "offline")
        try:
            return AgentStatus(status_str)
        except ValueError:
            return AgentStatus.OFFLINE

    @property
    def registered_at(self) -> Optional[datetime]:
        """Get registration time from filesystem."""
        ts = self._load_card().get("registered_at")
        if ts:
            try:
                return datetime.fromisoformat(ts)
            except ValueError:
                pass
        return None

    @property
    def last_heartbeat(self) -> datetime:
        """Get last heartbeat from filesystem.

        Returns epoch (1970-01-01) if not set, indicating no heartbeat received yet.
        """
        ts = self._load_card().get("last_heartbeat")
        if ts:
            try:
                return datetime.fromisoformat(ts)
            except ValueError:
                pass
        # Fallback to epoch for legacy cards without last_heartbeat
        return datetime(1970, 1, 1, tzinfo=UTC)

    @property
    def is_discoverable(self) -> bool:
        """Whether this participant appears in the public registry."""
        return self._load_card().get("discoverable_through_registry", True)

    def get_project(self, project_id: str):
        """Load a project by ID.

        Args:
            project_id: The project ID

        Returns:
            Project or None if not found
        """
        from .project import Project
        return Project.get(project_id, self._model_ctx)

    def exists(self) -> bool:
        """Check if participant exists on filesystem."""
        return self.card_path is not None

    # ─────────────────────────────────────────────────────────────────────────
    # Active Record Class Methods
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def get(cls, participant_id: str, ctx: "ModelContext") -> Optional["Self"]:
        """Load participant by ID from filesystem.

        Args:
            participant_id: The participant ID
            ctx: ModelContext for filesystem operations

        Returns:
            Participant instance or None if not found
        """
        instance = cls(participant_id, ctx)
        if not instance.exists():
            return None
        return instance

    @classmethod
    def get_or_create(cls, participant_id: str, ctx: "ModelContext") -> "Self":
        """Get existing participant or create a new instance.

        Unlike get(), this always returns an instance (never None).
        Use with update_card() for upsert operations.

        Args:
            participant_id: The participant's unique identifier
            ctx: ModelContext for filesystem access

        Returns:
            Existing or new participant instance
        """
        return cls(participant_id, ctx)

    @classmethod
    def get_by_name(cls, name: str, ctx: "ModelContext") -> Optional["Self"]:
        """Load participant by name (scans directory).

        Args:
            name: The participant name
            ctx: ModelContext for filesystem operations

        Returns:
            Participant instance or None if not found
        """
        if ctx.participants_dir is None:
            return None
        search_dir = ctx.participants_dir / cls._role_subdir
        if not search_dir.exists():
            return None
        name_lower = name.lower()
        for entry in search_dir.iterdir():
            if entry.is_dir():
                entry_name = entry.name.rsplit("-", 1)[0]
                if entry_name.lower() == name_lower:
                    data = FileUtil.read(entry / "card.json", "json")
                    if data:
                        return cls(data.get("id", ""), ctx)
        return None

    @classmethod
    def list_all(
        cls,
        ctx: "ModelContext",
        discoverable_only: bool = True,
        viewer_user_id: Optional[str] = None,
        viewer_is_admin: bool = False,
    ) -> list["Self"]:
        """List all participants from filesystem.

        Args:
            ctx: ModelContext for filesystem operations
            discoverable_only: If True, only return discoverable participants
                (plus non-discoverable ones owned by viewer or visible to admin)
            viewer_user_id: User ID of the viewer (for ownership-based visibility)
            viewer_is_admin: Whether the viewer is an admin (sees all agents)

        Returns:
            List of participant instances
        """
        result = []
        for entry in cls._list_dirs(ctx):
            data = FileUtil.read(entry / "card.json", "json")
            if data:
                is_discoverable = data.get("discoverable_through_registry", True)
                if viewer_is_admin:
                    # Admin sees all agents
                    result.append(cls(data["id"], ctx))
                elif viewer_user_id:
                    # Authenticated non-admin: own agents + discoverable agents
                    if data.get("registered_by") == viewer_user_id or is_discoverable:
                        result.append(cls(data["id"], ctx))
                elif not discoverable_only or is_discoverable:
                    # Unauthenticated: only discoverable agents
                    result.append(cls(data["id"], ctx))
        return result

    @classmethod
    def search(
        cls,
        ctx: "ModelContext",
        query: str = "",
        capabilities: Optional[list[str]] = None,
        status: Optional[str] = None,
        offset: int = 0,
        limit: int = 20,
        sort: str = "status_first",
        viewer_user_id: Optional[str] = None,
        viewer_is_admin: bool = False,
    ) -> tuple[list["Self"], int]:
        """Search participants with filtering and pagination.

        Loads card data once per agent and filters in-memory.

        Args:
            ctx: ModelContext for filesystem operations
            query: Text search across name and description (case-insensitive)
            capabilities: Filter by capabilities (agent must have at least one)
            status: Filter by agent status (e.g., "online")
            offset: Pagination offset
            limit: Page size (max 50)
            sort: Sort order - "status_first" (online first) or "name"
            viewer_user_id: User ID of the viewer (for visibility)
            viewer_is_admin: Whether the viewer is an admin

        Returns:
            Tuple of (paginated results, total matching count)
        """
        limit = min(limit, 50)
        query_lower = query.lower().strip()
        capabilities_set = set(c.lower() for c in (capabilities or []))

        # Load all cards once and filter
        matching: list[tuple[dict, "Self"]] = []
        for entry in cls._list_dirs(ctx):
            data = FileUtil.read(entry / "card.json", "json")
            if not data:
                continue

            # Visibility filter (same logic as list_all)
            is_discoverable = data.get("discoverable_through_registry", True)
            if not viewer_is_admin:
                if viewer_user_id:
                    if not (data.get("registered_by") == viewer_user_id or is_discoverable):
                        continue
                elif not is_discoverable:
                    continue

            # Text search filter
            if query_lower:
                name = data.get("name", "").lower()
                desc = data.get("description", "").lower()
                if query_lower not in name and query_lower not in desc:
                    continue

            # Capabilities filter
            if capabilities_set:
                agent_caps = set(c.lower() for c in data.get("capabilities", []))
                if not capabilities_set & agent_caps:
                    continue

            # Status filter
            if status:
                if data.get("status", "offline") != status:
                    continue

            matching.append((data, cls(data["id"], ctx)))

        # Sort
        if sort == "name":
            matching.sort(key=lambda x: x[0].get("name", ""))
        else:
            # status_first: online first, then verified, then alphabetical
            status_order = {"online": 0, "busy": 1, "rate_limited": 2, "offline": 3}
            matching.sort(key=lambda x: (
                status_order.get(x[0].get("status", "offline"), 3),
                0 if x[0].get("is_verified", False) else 1,
                x[0].get("name", ""),
            ))

        total = len(matching)
        page = [item[1] for item in matching[offset:offset + limit]]
        return page, total

    @classmethod
    def list_capabilities(
        cls,
        ctx: "ModelContext",
    ) -> list[str]:
        """Return distinct capabilities across all discoverable participants.

        Returns:
            Sorted list of unique capability strings
        """
        caps: set[str] = set()
        for entry in cls._list_dirs(ctx):
            data = FileUtil.read(entry / "card.json", "json")
            if data and data.get("discoverable_through_registry", True):
                for cap in data.get("capabilities", []):
                    caps.add(cap)
        return sorted(caps)

    @classmethod
    def _validate_name(cls, name: str) -> str:
        """Validate and normalize participant name.

        Args:
            name: Name to validate

        Returns:
            Normalized name (lowercase, filesystem-safe)

        Raises:
            ValueError: If name is invalid
        """
        return validate_name(name)

    @classmethod
    def register(
        cls,
        name: str,
        description: str,
        ctx: "ModelContext",
        discoverable: bool = True,
        capabilities: Optional[list[str]] = None,
        linked_user_id: Optional[str] = None,
        registered_by: Optional[str] = None,
    ) -> tuple["Self", str]:
        """Register a new participant. Returns (instance, token).

        Args:
            name: Participant name (no spaces)
            description: Participant description
            ctx: ModelContext for filesystem operations
            discoverable: Whether participant appears in public registry
            capabilities: List of capabilities
            linked_user_id: User ID to link (for assistants only)
            registered_by: User ID of the registrant

        Returns:
            Tuple of (participant instance, token)

        Raises:
            ValueError: If participant with same name already exists
        """
        # Validate name
        name = cls._validate_name(name)

        # Check uniqueness across all participant types
        from .participant import Participant

        if Participant.get_by_name(name, ctx) is not None:
            raise ValueError(f"Name '{name}' already registered")

        participant_id = secrets.token_hex(8)
        token = secrets.token_urlsafe(32)

        # Build card data
        # Initialize last_heartbeat to epoch (0) to indicate no heartbeat received yet
        epoch = datetime(1970, 1, 1, tzinfo=UTC)
        card_data = {
            "id": participant_id,
            "name": name,
            "description": description,
            "capabilities": capabilities or [],
            "status": AgentStatus.OFFLINE.value,
            "registered_at": datetime.now(UTC).isoformat(),
            "last_heartbeat": epoch.isoformat(),
            "discoverable_through_registry": discoverable,
            "registered_by": registered_by,
            "is_verified": False,
        }

        # Add linked_user_id for assistants
        if linked_user_id is not None:
            card_data["linked_user_id"] = linked_user_id

        # Save to filesystem (directories created by FileUtil.write)
        part_dir = cls._get_dir(ctx, name, participant_id)
        FileUtil.write(part_dir / "card.json", card_data, "json", atomic=True)
        FileUtil.write(part_dir / "credential.json", {
            "token_hash": hashlib.sha256(token.encode()).hexdigest()
        }, "json", atomic=True)

        return cls(participant_id, ctx), token

    def regenerate_token(self) -> str:
        """Generate a new token and update credential.json.

        Returns:
            The new raw token (only available at call time).
        """
        token = secrets.token_urlsafe(32)
        part_dir = self._get_dir(self._model_ctx, self.name, self._id)
        FileUtil.write(part_dir / "credential.json", {
            "token_hash": hashlib.sha256(token.encode()).hexdigest()
        }, "json", atomic=True)
        return token

    @classmethod
    def verify_token(cls, participant_id: str, token: str, ctx: "ModelContext") -> bool:
        """Verify participant token.

        Args:
            participant_id: The participant ID
            token: The token to verify
            ctx: ModelContext for filesystem operations

        Returns:
            True if token is valid
        """
        # Use cls._role_subdir directly instead of scanning all subdirs
        search_dir = ctx.participants_dir / cls._role_subdir
        if not search_dir.exists():
            return False
        matches = list(search_dir.glob(f"*-{participant_id}"))
        if not matches:
            return False
        cred_path = matches[0] / "credential.json"
        if not cred_path.exists():
            return False
        data = FileUtil.read(cred_path, "json")
        if not data:
            return False
        stored_hash = data.get("token_hash", "")
        return secrets.compare_digest(
            stored_hash,
            hashlib.sha256(token.encode()).hexdigest()
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Instance Methods
    # ─────────────────────────────────────────────────────────────────────────

    def save(self) -> None:
        """Save current state to filesystem."""
        card = self._load_card()
        self._save_card(card)

    @property
    def registered_by(self) -> Optional[str]:
        """Get the user ID of who registered this participant."""
        return self._load_card().get("registered_by")

    @property
    def is_verified(self) -> bool:
        """Check if this participant is admin-verified."""
        return self._load_card().get("is_verified", False)

    def verify(self) -> None:
        """Mark this participant as verified."""
        card = self._load_card()
        card["is_verified"] = True
        self._save_card(card)

    def unverify(self) -> None:
        """Remove verification from this participant."""
        card = self._load_card()
        card["is_verified"] = False
        self._save_card(card)

    def update_status(self, status: AgentStatus) -> None:
        """Update status and save.

        Args:
            status: New status to set
        """
        card = self._load_card()
        card["status"] = status.value
        self._save_card(card)

    def heartbeat(self) -> None:
        """Update heartbeat timestamp and set ONLINE status."""
        card = self._load_card()
        card["status"] = AgentStatus.ONLINE.value
        card["last_heartbeat"] = datetime.now(UTC).isoformat()
        self._save_card(card)

    def update_card(self, **fields) -> None:
        """Update specific fields in card.json and save.

        Works for both existing and new participants (upsert semantics):
        - For existing: loads card, merges fields, saves
        - For new: starts with empty dict, merges fields, saves (creates dir)

        For new participants, `name` must be in fields for proper directory naming.

        Args:
            **fields: Fields to update in the card (e.g., status="online")
        """
        card = self._load_card()  # Returns {} for new participants
        card.update(fields)
        self._save_card(card)  # Creates directory via ensure_participant_dir()

    def to_response(self) -> "AgentResponse":
        """Convert to API response model.

        Returns:
            AgentResponse DTO for API serialization
        """
        return AgentResponse(
            id=self._id,
            name=self.name,
            description=self.description,
            capabilities=self.capabilities,
            status=self.status,
            registered_at=self.registered_at or datetime.now(UTC),
            last_heartbeat=self.last_heartbeat,
            discoverable_through_registry=self.is_discoverable,
            registered_by=self.registered_by,
            is_verified=self.is_verified,
        )

    def to_dict(self) -> dict:
        """Serialize to dictionary (reads from filesystem)."""
        card = self._load_card()
        return {
            "id": self._id,
            "name": card.get("name", ""),
            "description": card.get("description", ""),
            "capabilities": card.get("capabilities", []),
            "status": card.get("status", "offline"),
            "registered_at": card.get("registered_at", ""),
            "last_heartbeat": card.get("last_heartbeat"),
            "discoverable_through_registry": card.get("discoverable_through_registry", True),
            "registered_by": card.get("registered_by"),
            "is_verified": card.get("is_verified", False),
            "role": self.role.value,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self._id!r}, name={self.name!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PersistableParticipant):
            return NotImplemented
        return self._id == other._id

    def __hash__(self) -> int:
        return hash(self._id)
