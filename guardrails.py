"""Deterministic input guardrails for the Cred support agent."""

import re
from dataclasses import dataclass


PAN_RE = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", re.IGNORECASE)
AADHAAR_RE = re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")
BANK_CONTEXT_RE = re.compile(
    r"(?i)\b(?:bank\s+account|account\s+number|a/?c)\s*"
    r"(?:number|no\.?|#)?\s*[:=-]?\s*(\d{9,18})\b"
)

INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior|system)\s+instructions", re.I),
    re.compile(r"reveal\s+(?:the\s+)?(?:system|developer)\s+prompt", re.I),
    re.compile(r"show\s+(?:me\s+)?(?:your\s+)?hidden\s+instructions", re.I),
    re.compile(r"bypass\s+(?:the\s+)?(?:guardrail|safety|policy)", re.I),
    re.compile(r"act\s+as\s+if\s+(?:the\s+)?rules\s+do\s+not\s+apply", re.I),
]


@dataclass(frozen=True)
class GuardrailResult:
    text: str
    pii_masked: bool
    injection_detected: bool


def mask_pii(text: str) -> str:
    """
    Mask fixed-format PII and return the masked string.

    This keeps compatibility with the starter project's original tests.
    """
    masked = PAN_RE.sub("[MASKED_PAN]", text)
    masked = AADHAAR_RE.sub("[MASKED_AADHAAR]", masked)

    def mask_bank(match: re.Match) -> str:
        full_match = match.group(0)
        account_number = match.group(1)
        prefix_end = full_match.find(account_number)
        prefix = full_match[:prefix_end]
        return f"{prefix}[MASKED_BANK_ACCOUNT]"

    masked = BANK_CONTEXT_RE.sub(mask_bank, masked)
    return masked


def mask_pii_with_status(text: str) -> tuple[str, bool]:
    """Return masked text plus a flag showing whether any masking occurred."""
    masked = mask_pii(text)
    return masked, masked != text


def detect_prompt_injection(text: str) -> bool:
    return any(pattern.search(text) for pattern in INJECTION_PATTERNS)


def apply_input_guardrails(text: str) -> GuardrailResult:
    masked, changed = mask_pii_with_status(text)

    return GuardrailResult(
        text=masked,
        pii_masked=changed,
        injection_detected=detect_prompt_injection(text),
    )


def check_input(text: str) -> str:
    """
    Backward-compatible starter helper.

    Raises ValueError for prompt-injection attempts and otherwise returns
    the PII-masked text.
    """
    result = apply_input_guardrails(text)

    if result.injection_detected:
        raise ValueError("Prompt-injection pattern detected")

    return result.text
