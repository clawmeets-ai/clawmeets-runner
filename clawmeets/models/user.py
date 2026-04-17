# SPDX-License-Identifier: MIT
"""
clawmeets/models/user.py
Human user implementation with Active Record persistence.

Users create projects and communicate with their assistant.
They don't auto-respond to messages - interaction happens via UI.

All state is read from the passwd file - no in-memory state.
"""
from __future__ import annotations

import asyncio
import json
import logging
import secrets
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import bcrypt

from .participant import Participant, ParticipantRole
from clawmeets.utils.file_io import FileUtil
from clawmeets.utils.validation import validate_name

if TYPE_CHECKING:
    from ..api.client import ClawMeetsClient
    from .context import ModelContext
    from .chat_message import ChatMessage

logger = logging.getLogger("clawmeets.user")


# Module-level lock for passwd file operations
_passwd_lock = asyncio.Lock()


@dataclass
class NotificationConfig:
    """Configuration for user notification scripts."""
    script_path: str
    timeout: float = 30.0
    fail_fast: bool = False


def _now() -> datetime:
    return datetime.now(UTC)


class User(Participant):
    """
    Human user - creates projects, communicates with assistant.

    Users interact via the UI/API, not via automated responses.
    The on_message handler is primarily for UI notifications.

    All state is read from the passwd file on each property access.
    This ensures the model always reflects the current state on disk.
    """

    def __init__(
        self,
        id: str,
        model_ctx: "ModelContext",
    ) -> None:
        """Initialize a User.

        Args:
            id: The user's unique identifier
            model_ctx: ModelContext for filesystem access (may include client)
        """
        super().__init__()
        self._id = id
        self._model_ctx = model_ctx
        self._notification_config: Optional[NotificationConfig] = None

    # ─────────────────────────────────────────────────────────────────────────
    # Static Helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _passwd_path(ctx: "ModelContext") -> Path:
        """Get path to passwd file.

        Args:
            ctx: ModelContext for path resolution

        Returns:
            Path to passwd file
        """
        return ctx.participants_dir / "passwd"

    @staticmethod
    def _validate_username(username: str) -> str:
        """Validate and normalize username.

        Args:
            username: The username to validate

        Returns:
            Normalized username (lowercase, stripped)

        Raises:
            ValueError: If username is invalid
        """
        username = validate_name(username)
        if "-" in username:
            raise ValueError("Username cannot contain hyphens (-). Use underscores (_) instead.")
        return username

    @staticmethod
    def _hash_password(password: str) -> str:
        """Hash a password using bcrypt."""
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    @staticmethod
    def _verify_password_hash(password: str, password_hash: str) -> bool:
        """Verify a password against its hash."""
        try:
            return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
        except Exception:
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # Passwd File I/O
    # ─────────────────────────────────────────────────────────────────────────

    def _load_passwd(self) -> dict:
        """Load the passwd file, returning empty structure if doesn't exist."""
        passwd_path = self._passwd_path(self._model_ctx)
        data = FileUtil.read(passwd_path, "json", default=None)
        return data if data is not None else {"users": {}}

    def _save_passwd(self, data: dict) -> None:
        """Save the passwd file atomically."""
        passwd_path = self._passwd_path(self._model_ctx)
        FileUtil.write(passwd_path, data, "json", atomic=True)

    def _load_passwd_entry(self) -> dict:
        """Load this user's entry from passwd file.

        Returns:
            Dict of user data, or empty dict if not found
        """
        data = self._load_passwd()
        users = data.get("users", {})
        return users.get(self._id, {})

    # ─────────────────────────────────────────────────────────────────────────
    # Active Record Class Methods
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def get(cls, user_id: str, ctx: "ModelContext") -> Optional["User"]:
        """Load user by ID from passwd file.

        Args:
            user_id: The user ID
            ctx: ModelContext for filesystem operations

        Returns:
            User instance or None if not found
        """
        user = cls(user_id, ctx)
        if not user.exists():
            return None
        return user

    @classmethod
    def get_by_username(cls, username: str, ctx: "ModelContext") -> Optional["User"]:
        """Load user by username.

        Args:
            username: The username to find
            ctx: ModelContext for filesystem operations

        Returns:
            User instance or None if not found
        """
        username = username.lower().strip()
        passwd_path = cls._passwd_path(ctx)
        data = FileUtil.read(passwd_path, "json", default=None)
        if data is None:
            return None

        users = data.get("users", {})
        for user_id, user_data in users.items():
            if user_data.get("username") == username:
                return cls(user_id, ctx)
        return None

    @classmethod
    def list_all(cls, ctx: "ModelContext") -> list["User"]:
        """List all users.

        Args:
            ctx: ModelContext for filesystem operations

        Returns:
            List of User instances
        """
        passwd_path = cls._passwd_path(ctx)
        data = FileUtil.read(passwd_path, "json", default=None)
        if data is None:
            return []

        users = data.get("users", {})
        return [cls(user_id, ctx) for user_id in users.keys()]

    @classmethod
    def get_by_email(cls, email: str, ctx: "ModelContext") -> Optional["User"]:
        """Load user by email address (case-insensitive).

        Args:
            email: The email to find
            ctx: ModelContext for filesystem operations

        Returns:
            User instance or None if not found
        """
        email = email.strip().lower()
        passwd_path = cls._passwd_path(ctx)
        data = FileUtil.read(passwd_path, "json", default=None)
        if data is None:
            return None

        users = data.get("users", {})
        for user_id, user_data in users.items():
            if user_data.get("email", "").lower() == email:
                return cls(user_id, ctx)
        return None

    @classmethod
    def get_by_phone(cls, phone_number: str, ctx: "ModelContext") -> Optional["User"]:
        """Load user by phone number.

        Args:
            phone_number: The phone number to find
            ctx: ModelContext for filesystem operations

        Returns:
            User instance or None if not found
        """
        phone_number = phone_number.strip()
        passwd_path = cls._passwd_path(ctx)
        data = FileUtil.read(passwd_path, "json", default=None)
        if data is None:
            return None

        users = data.get("users", {})
        for user_id, user_data in users.items():
            if user_data.get("phone_number") == phone_number:
                return cls(user_id, ctx)
        return None

    @classmethod
    async def verify_email(cls, token: str, ctx: "ModelContext") -> Optional["User"]:
        """Verify a user's email using the verification token.

        Args:
            token: The verification token from the email link
            ctx: ModelContext for filesystem operations

        Returns:
            User instance if token is valid, None otherwise
        """
        async with _passwd_lock:
            passwd_path = cls._passwd_path(ctx)
            data = FileUtil.read(passwd_path, "json", default=None)
            if data is None:
                return None

            users = data.get("users", {})
            for user_id, user_data in users.items():
                if user_data.get("verification_token") == token:
                    user_data["email_verified"] = True
                    user_data["verification_token"] = None
                    data["users"] = users
                    FileUtil.write(passwd_path, data, "json", atomic=True)
                    return cls(user_id, ctx)

            return None

    @classmethod
    async def register(
        cls,
        username: str,
        password: str,
        ctx: "ModelContext",
        is_admin: bool = False,
        email: Optional[str] = None,
        email_verified: bool = False,
    ) -> "User":
        """Create a new user.

        Args:
            username: The username (will be normalized)
            password: The password (will be hashed)
            ctx: ModelContext for filesystem operations
            is_admin: Whether user is an admin
            email: Email address (required for self-registration)
            email_verified: Whether email is pre-verified (True for admin-created users)

        Returns:
            New User instance

        Raises:
            ValueError: If username is invalid or already exists, or email already registered
        """
        username = cls._validate_username(username)

        async with _passwd_lock:
            passwd_path = cls._passwd_path(ctx)

            # Load existing data
            data = FileUtil.read(passwd_path, "json", default=None)
            if data is None:
                data = {"users": {}}

            users = data.get("users", {})

            # Check username uniqueness across all participant types
            from .participant import Participant

            if Participant.get_by_name(username, ctx) is not None:
                raise ValueError(f"Username '{username}' already registered")

            # Check email uniqueness (if email provided)
            if email:
                email = email.strip().lower()
                for user_data in users.values():
                    if user_data.get("email", "").lower() == email:
                        raise ValueError(f"Email '{email}' already registered")

            user_id = str(uuid.uuid4())
            entry = {
                "id": user_id,
                "username": username,
                "role": "admin" if is_admin else "user",
                "password_hash": cls._hash_password(password),
                "created_at": _now().isoformat(),
                "email": email,
                "email_verified": email_verified,
                "verification_token": None if email_verified else secrets.token_urlsafe(32),
            }
            users[user_id] = entry
            data["users"] = users

            # Save atomically
            FileUtil.write(passwd_path, data, "json", atomic=True)

            return cls(user_id, ctx)

    @classmethod
    async def verify_password(
        cls,
        username: str,
        password: str,
        ctx: "ModelContext",
    ) -> Optional["User"]:
        """Verify username/password and return user if valid.

        Args:
            username: The username
            password: The password to verify
            ctx: ModelContext for filesystem operations

        Returns:
            User instance if valid, None otherwise
        """
        username = username.lower().strip()
        passwd_path = cls._passwd_path(ctx)
        data = FileUtil.read(passwd_path, "json", default=None)
        if data is None:
            return None

        users = data.get("users", {})
        for user_id, user_data in users.items():
            if user_data.get("username") == username:
                if cls._verify_password_hash(password, user_data.get("password_hash", "")):
                    return cls(user_id, ctx)
                return None  # Wrong password
        return None  # User not found

    @classmethod
    async def initialize(cls, ctx: "ModelContext") -> None:
        """Initialize user store, ensuring passwd file exists.

        If no admin user is found, logs a warning instructing the operator
        to create one via `admin init-passwd`.

        Args:
            ctx: ModelContext for filesystem operations
        """
        async with _passwd_lock:
            passwd_path = cls._passwd_path(ctx)

            # Load existing data
            data = FileUtil.read(passwd_path, "json", default=None)
            if data is None:
                data = {"users": {}}
                FileUtil.write(passwd_path, data, "json", atomic=True)

            users = data.get("users", {})

            # Check if any admin user exists
            has_admin = any(
                u.get("role") == "admin"
                for u in users.values()
            )

            if not has_admin:
                logger.warning(
                    "No admin user found. Create one with: "
                    "python -m clawmeets.cli admin init-passwd <password>"
                )

    # ─────────────────────────────────────────────────────────────────────────
    # Instance Properties
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        """Get username from filesystem."""
        return self._load_passwd_entry().get("username", "")

    @property
    def username(self) -> str:
        """Alias for name - users have usernames."""
        return self.name

    @property
    def role(self) -> ParticipantRole:
        return ParticipantRole.USER

    @property
    def assistant_id(self) -> Optional[str]:
        """The ID of this user's linked assistant (from filesystem)."""
        return self._load_passwd_entry().get("assistant_agent_id")

    @property
    def created_at(self) -> Optional[datetime]:
        """Get creation time from filesystem."""
        ts = self._load_passwd_entry().get("created_at")
        if ts:
            try:
                return datetime.fromisoformat(ts)
            except ValueError:
                pass
        return None

    @property
    def has_assistant(self) -> bool:
        """Check if user has a linked assistant."""
        return self.assistant_id is not None

    @property
    def is_admin(self) -> bool:
        """Check if user is an admin."""
        return self._load_passwd_entry().get("role") == "admin"

    @property
    def email(self) -> Optional[str]:
        """Get email from filesystem."""
        return self._load_passwd_entry().get("email")

    @property
    def email_verified(self) -> bool:
        """Check if user's email is verified."""
        return self._load_passwd_entry().get("email_verified", False)

    @property
    def verification_token(self) -> Optional[str]:
        """Get verification token (None if already verified)."""
        return self._load_passwd_entry().get("verification_token")

    @property
    def phone_number(self) -> Optional[str]:
        """Get phone number from filesystem."""
        return self._load_passwd_entry().get("phone_number")

    @property
    def phone_verified(self) -> bool:
        """Check if user's phone number is verified."""
        return self._load_passwd_entry().get("phone_verified", False)

    @property
    def phone_verification_code(self) -> Optional[str]:
        """Get phone verification code (None if already verified)."""
        return self._load_passwd_entry().get("phone_verification_code")

    @property
    def description(self) -> str:
        """Users don't have descriptions."""
        return ""

    @property
    def user_role(self) -> str:
        """Get the user's role (admin/user)."""
        return self._load_passwd_entry().get("role", "user")

    def get_project(self, project_id: str):
        """Load a project by ID.

        Args:
            project_id: The project ID

        Returns:
            Project or None if not found
        """
        from .project import Project
        return Project.get(project_id, self._model_ctx)

    def set_notification_config(self, config: NotificationConfig) -> None:
        """Set notification configuration for message callbacks.

        Args:
            config: NotificationConfig with script path and settings
        """
        self._notification_config = config

    def exists(self) -> bool:
        """Check if user exists in passwd file."""
        return bool(self._load_passwd_entry())

    def update_card(self, **fields) -> None:
        """No-op for User - users don't have card.json."""
        pass

    # ─────────────────────────────────────────────────────────────────────────
    # Instance Methods
    # ─────────────────────────────────────────────────────────────────────────

    async def change_password(self, new_password: str) -> None:
        """Change this user's password.

        Args:
            new_password: The new password
        """
        async with _passwd_lock:
            data = self._load_passwd()
            users = data.get("users", {})
            if self._id not in users:
                raise KeyError(f"User {self._id!r} not found")
            users[self._id]["password_hash"] = self._hash_password(new_password)
            data["users"] = users
            self._save_passwd(data)

    async def set_email_verified(self, verified: bool) -> None:
        """Set email verification status.

        Args:
            verified: Whether email is verified
        """
        async with _passwd_lock:
            data = self._load_passwd()
            users = data.get("users", {})
            if self._id not in users:
                raise KeyError(f"User {self._id!r} not found")
            users[self._id]["email_verified"] = verified
            if verified:
                users[self._id]["verification_token"] = None
            data["users"] = users
            self._save_passwd(data)

    async def update_phone_number(self, phone_number: str, verification_code: str) -> None:
        """Update phone number and set verification code.

        Args:
            phone_number: The new phone number (E.164 format)
            verification_code: The verification code to send via SMS
        """
        async with _passwd_lock:
            data = self._load_passwd()
            users = data.get("users", {})
            if self._id not in users:
                raise KeyError(f"User {self._id!r} not found")
            users[self._id]["phone_number"] = phone_number
            users[self._id]["phone_verified"] = False
            users[self._id]["phone_verification_code"] = verification_code
            data["users"] = users
            self._save_passwd(data)

    async def set_phone_verified(self, verified: bool) -> None:
        """Set phone verification status.

        Args:
            verified: Whether phone is verified
        """
        async with _passwd_lock:
            data = self._load_passwd()
            users = data.get("users", {})
            if self._id not in users:
                raise KeyError(f"User {self._id!r} not found")
            users[self._id]["phone_verified"] = verified
            if verified:
                users[self._id]["phone_verification_code"] = None
            data["users"] = users
            self._save_passwd(data)

    async def delete(self) -> None:
        """Delete this user."""
        async with _passwd_lock:
            data = self._load_passwd()
            users = data.get("users", {})
            if self._id not in users:
                raise KeyError(f"User {self._id!r} not found")
            del users[self._id]
            data["users"] = users
            self._save_passwd(data)

    async def link_assistant(self, assistant_agent_id: str) -> None:
        """Link an assistant agent to this user.

        Args:
            assistant_agent_id: The assistant agent's ID
        """
        async with _passwd_lock:
            data = self._load_passwd()
            users = data.get("users", {})
            if self._id not in users:
                raise KeyError(f"User {self._id!r} not found")
            users[self._id]["assistant_agent_id"] = assistant_agent_id
            data["users"] = users
            self._save_passwd(data)

    # ─────────────────────────────────────────────────────────────────────────
    # Event Handlers
    # ─────────────────────────────────────────────────────────────────────────

    async def on_message(
        self,
        project_id: str,
        chatroom_name: str,
        message: "ChatMessage",
        addressed_to_me: bool,
    ) -> None:
        """
        User receives messages - triggers notification script if configured.

        Filtering:
        - Skip if no notification config
        - Skip acknowledgment messages (is_ack=True)
        - Skip non-user-communication chatrooms
        - Skip messages not from the project coordinator
        """
        from .chatroom import Chatroom
        from .project import Project

        # Skip if no notification config
        if not self._notification_config:
            return

        # Skip acknowledgment messages
        if message.is_ack:
            return

        # Skip non-user-communication chatrooms
        chatroom = Chatroom.get(project_id, chatroom_name, self._model_ctx)
        if not chatroom or not chatroom.is_user_communication_room:
            return

        # Skip messages not from coordinator
        project = Project.get(project_id, self._model_ctx)
        if not project or message.from_participant_id != project.coordinator_id:
            return

        # Build notification payload
        payload = {
            "event": "message",
            "project_id": project_id,
            "project_name": project.name,
            "chatroom_name": chatroom_name,
            "user_id": self._id,
            "username": self.username,
            "message": {
                "id": message.id,
                "ts": message.ts.isoformat() if message.ts else None,
                "from_participant_id": message.from_participant_id,
                "from_participant_name": message.from_participant_name,
                "content": message.content,
            },
        }

        await self._execute_notification_script(payload)

    async def _execute_notification_script(self, payload: dict) -> None:
        """Execute the configured notification script with payload via stdin.

        Args:
            payload: JSON-serializable notification payload
        """
        if not self._notification_config:
            return

        try:
            proc = await asyncio.create_subprocess_exec(
                self._notification_config.script_path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(input=json.dumps(payload).encode()),
                    timeout=self._notification_config.timeout,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                logger.error(
                    f"Notification script timed out after {self._notification_config.timeout}s"
                )
                if self._notification_config.fail_fast:
                    raise RuntimeError("Notification script timed out")
                return

            if proc.returncode != 0:
                logger.error(
                    f"Notification script exited with code {proc.returncode}: "
                    f"stderr={stderr.decode()}"
                )
                if self._notification_config.fail_fast:
                    raise RuntimeError(
                        f"Notification script failed with code {proc.returncode}"
                    )
            else:
                if stdout:
                    logger.debug(f"Notification script output: {stdout.decode()}")

        except FileNotFoundError:
            logger.error(
                f"Notification script not found: {self._notification_config.script_path}"
            )
            if self._notification_config.fail_fast:
                raise
        except Exception as e:
            logger.error(f"Failed to execute notification script: {e}", exc_info=True)
            if self._notification_config.fail_fast:
                raise

    async def on_project_completed(
        self,
        project_id: str,
    ) -> None:
        """
        Notification that a project the user created has completed.
        Can be used to trigger UI notifications.
        """
        # Could emit WebSocket notification here
        pass

    # ─────────────────────────────────────────────────────────────────────────
    # Serialization
    # ─────────────────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize to dictionary (reads from filesystem)."""
        entry = self._load_passwd_entry()
        return {
            "id": self._id,
            "username": entry.get("username", ""),
            "email": entry.get("email"),
            "email_verified": entry.get("email_verified", False),
            "phone_number": entry.get("phone_number"),
            "phone_verified": entry.get("phone_verified", False),
            "assistant_agent_id": entry.get("assistant_agent_id"),
            "created_at": entry.get("created_at", ""),
            "is_admin": entry.get("role") == "admin",
        }

    def __repr__(self) -> str:
        return f"User(id={self._id!r}, username={self.username!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, User):
            return NotImplemented
        return self._id == other._id

    def __hash__(self) -> int:
        return hash(self._id)
