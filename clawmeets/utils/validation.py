# SPDX-License-Identifier: MIT
"""
clawmeets/utils/validation.py
Shared validation utilities for filesystem-compatible names.

Provides validation for user, agent, project, and chatroom names to ensure
they are safe for filesystem paths and ASCII-only.
"""
import re

# Pattern: alphanumeric (case-sensitive), may contain - or _ but not at start/end or consecutive
# Matches: myproject, MyProject, my-project, My_Project, project-123, a
# Rejects: -project, project-, project--name, my project
# Structure: start with alnum, then (optional separator + alnum)* - ensures no consecutive separators
NAME_PATTERN = re.compile(r"^[a-zA-Z0-9]([_-]?[a-zA-Z0-9])*$")

# Windows reserved names (checked case-insensitively)
RESERVED_NAMES = frozenset([
    "con", "prn", "aux", "nul",
    "com1", "com2", "com3", "com4", "com5", "com6", "com7", "com8", "com9",
    "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9",
])

# Maximum allowed length for names
MAX_NAME_LENGTH = 64


# Email pattern: standard format validation
EMAIL_PATTERN = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")


def validate_email(email: str) -> str:
    """Validate and normalize an email address.

    Args:
        email: The email to validate

    Returns:
        Normalized email (lowercase, stripped)

    Raises:
        ValueError: If email format is invalid
    """
    email = email.strip().lower()
    if not email:
        raise ValueError("Email cannot be empty")
    if len(email) > 254:
        raise ValueError("Email address too long")
    if not EMAIL_PATTERN.match(email):
        raise ValueError(f"Invalid email format: {email}")
    return email


def validate_name(name: str) -> str:
    """Validate a name for filesystem compatibility (case-sensitive).

    A valid name must:
    1. Be non-empty
    2. Be ASCII-only (a-z, A-Z, 0-9, and allowed separators)
    3. Start with a letter or number (not a separator)
    4. Not end with a separator
    5. Use only allowed separators: - (hyphen), _ (underscore)
    6. No consecutive separators
    7. No reserved names or characters problematic for filesystems
    8. No leading or trailing whitespace

    Args:
        name: The name to validate

    Returns:
        The original name (unmodified) if valid

    Raises:
        ValueError: If name is invalid
    """
    if not name:
        raise ValueError("name cannot be empty")

    if name != name.strip():
        raise ValueError("name cannot have leading or trailing whitespace")

    if len(name) > MAX_NAME_LENGTH:
        raise ValueError(f"name cannot exceed {MAX_NAME_LENGTH} characters")

    if not name.isascii():
        raise ValueError("name must contain only ASCII characters")

    if " " in name:
        raise ValueError("name cannot contain spaces")

    if not NAME_PATTERN.match(name):
        if name.startswith(("-", "_")):
            raise ValueError("name cannot start with a separator (- or _)")
        if name.endswith(("-", "_")):
            raise ValueError("name cannot end with a separator (- or _)")
        if "--" in name or "__" in name or "-_" in name or "_-" in name:
            raise ValueError("name cannot contain consecutive separators")
        if name.startswith("."):
            raise ValueError("name cannot start with a dot")
        raise ValueError(
            "name must contain only letters, numbers, hyphens, and underscores"
        )

    if name.lower() in RESERVED_NAMES:
        raise ValueError(f"name cannot be a reserved name: {name}")

    return name
