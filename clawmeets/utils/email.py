# SPDX-License-Identifier: MIT
"""
clawmeets/utils/email.py
Email sending utility with SendGrid support and console fallback.
"""
import logging
import os

logger = logging.getLogger("clawmeets.email")

SENDGRID_API_KEY = os.environ.get("SENDGRID_API_KEY")
SENDGRID_FROM_EMAIL = os.environ.get("SENDGRID_FROM_EMAIL", "info@clawmeets.ai")
VERIFICATION_BASE_URL = os.environ.get(
    "CLAWMEETS_VERIFICATION_URL", "https://clawmeets.ai"
)


async def send_verification_email(
    to_email: str,
    username: str,
    verification_token: str,
) -> None:
    """Send email verification link.

    Uses SendGrid if SENDGRID_API_KEY is set, otherwise logs to console.
    """
    verification_url = (
        f"{VERIFICATION_BASE_URL}/auth/verify-email?token={verification_token}"
    )

    if SENDGRID_API_KEY:
        await _send_via_sendgrid(to_email, username, verification_url)
    else:
        _send_via_console(to_email, username, verification_url)


async def send_waitlist_email(to_email: str) -> None:
    """Send waitlist confirmation email, BCC chengtao@clawmeets.ai.

    Uses SendGrid if SENDGRID_API_KEY is set, otherwise logs to console.
    """
    if SENDGRID_API_KEY:
        await _send_waitlist_via_sendgrid(to_email)
    else:
        _send_waitlist_via_console(to_email)


async def _send_waitlist_via_sendgrid(to_email: str) -> None:
    """Send waitlist confirmation via SendGrid API."""
    try:
        import sendgrid
        from sendgrid.helpers.mail import Mail, Email, To, Content, Bcc
    except ImportError:
        logger.error(
            "SendGrid is not installed. Install with: pip install sendgrid\n"
            "Falling back to console output."
        )
        _send_waitlist_via_console(to_email)
        return

    sg = sendgrid.SendGridAPIClient(api_key=SENDGRID_API_KEY)
    message = Mail(
        from_email=Email(SENDGRID_FROM_EMAIL),
        to_emails=To(to_email),
        subject="You're on the ClawMeets waitlist",
        plain_text_content=Content(
            "text/plain",
            f"Thank you for your interest in ClawMeets!\n\n"
            f"You're now on our waitlist. We'll send you an invitation code as soon as possible.\n\n"
            f"If you have a strong use case and would like to expedite access, "
            f"please reply to this email or reach out to info@clawmeets.ai.\n\n"
            f"— The ClawMeets Team\n",
        ),
    )
    message.add_bcc(Bcc("chengtao@clawmeets.ai"))
    try:
        response = sg.send(message)
        logger.info(
            f"Waitlist email sent to {to_email} "
            f"(status={response.status_code})"
        )
    except Exception as e:
        logger.error(f"Failed to send waitlist email to {to_email}: {e}")
        raise


def _send_waitlist_via_console(to_email: str) -> None:
    """Fallback: log waitlist email to console."""
    logger.info(
        f"[EMAIL FALLBACK] Waitlist email for {to_email}"
    )
    print(f"\n--- Waitlist Email (SendGrid not configured) ---")
    print(f"To: {to_email}")
    print(f"BCC: chengtao@clawmeets.ai")
    print(f"Subject: You're on the ClawMeets waitlist")
    print(f"Body: Thank you for your interest! We'll send an invitation code ASAP.")
    print(f"------------------------------------------------\n")


async def _send_via_sendgrid(
    to_email: str,
    username: str,
    verification_url: str,
) -> None:
    """Send via SendGrid API."""
    try:
        import sendgrid
        from sendgrid.helpers.mail import Mail, Email, To, Content
    except ImportError:
        logger.error(
            "SendGrid is not installed. Install with: pip install sendgrid\n"
            "Falling back to console output."
        )
        _send_via_console(to_email, username, verification_url)
        return

    sg = sendgrid.SendGridAPIClient(api_key=SENDGRID_API_KEY)
    message = Mail(
        from_email=Email(SENDGRID_FROM_EMAIL),
        to_emails=To(to_email),
        subject="Verify your ClawMeets account",
        plain_text_content=Content(
            "text/plain",
            f"Hello {username},\n\n"
            f"Click the link below to verify your email:\n\n"
            f"{verification_url}\n\n"
            f"This link will expire when used.\n",
        ),
    )
    try:
        response = sg.send(message)
        logger.info(
            f"Verification email sent to {to_email} "
            f"(status={response.status_code})"
        )
    except Exception as e:
        logger.error(f"Failed to send verification email to {to_email}: {e}")
        raise


def _send_via_console(
    to_email: str,
    username: str,
    verification_url: str,
) -> None:
    """Fallback: log verification link to console."""
    logger.info(
        f"[EMAIL FALLBACK] Verification email for {username} ({to_email}):\n"
        f"  Verify at: {verification_url}"
    )
    print(f"\n--- Verification Email (SendGrid not configured) ---")
    print(f"To: {to_email}")
    print(f"Subject: Verify your ClawMeets account")
    print(f"Verify at: {verification_url}")
    print(f"---------------------------------------------------\n")
