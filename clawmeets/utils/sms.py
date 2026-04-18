# SPDX-License-Identifier: MIT
"""
clawmeets/utils/sms.py
SMS sending utility with Twilio support and console fallback.
"""
import logging
import os

logger = logging.getLogger("clawmeets.sms")

TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER = os.environ.get("TWILIO_FROM_NUMBER")


async def send_verification_sms(
    to_phone: str,
    code: str,
) -> None:
    """Send SMS with phone verification code.

    Uses Twilio if TWILIO_ACCOUNT_SID is set, otherwise logs to console.
    """
    body = f"Your ClawMeets verification code is: {code}"
    if TWILIO_ACCOUNT_SID:
        await _send_via_twilio(to_phone, body)
    else:
        _send_via_console(to_phone, body)


async def send_notification_sms(
    to_phone: str,
    message: str,
) -> None:
    """Send a notification SMS.

    Uses Twilio if TWILIO_ACCOUNT_SID is set, otherwise logs to console.
    """
    if TWILIO_ACCOUNT_SID:
        await _send_via_twilio(to_phone, message)
    else:
        _send_via_console(to_phone, message)


async def _send_via_twilio(to_phone: str, body: str) -> None:
    """Send SMS via Twilio API."""
    try:
        from twilio.rest import Client
    except ImportError:
        logger.error(
            "Twilio is not installed. Install with: pip install 'clawmeets[sms]'\n"
            "Falling back to console output."
        )
        _send_via_console(to_phone, body)
        return

    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    try:
        message = client.messages.create(
            body=body,
            from_=TWILIO_FROM_NUMBER,
            to=to_phone,
        )
        logger.info(f"SMS sent to {to_phone} (sid={message.sid})")
    except Exception as e:
        logger.error(f"Failed to send SMS to {to_phone}: {e}")
        raise


def _send_via_console(to_phone: str, body: str) -> None:
    """Fallback: log SMS to console."""
    logger.info(f"[SMS FALLBACK] SMS for {to_phone}")
    print(f"\n--- SMS (Twilio not configured) ---")
    print(f"To: {to_phone}")
    print(f"Message: {body}")
    print(f"-----------------------------------\n")
