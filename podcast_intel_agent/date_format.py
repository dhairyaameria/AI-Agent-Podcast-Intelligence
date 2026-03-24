"""Human-readable dates for briefing output (ISO / RSS strings in → readable out)."""

from __future__ import annotations

from datetime import datetime, timezone


def format_published_for_briefing(raw: str | None) -> str:
    """e.g. ``2026-03-20T16:00:55+00:00`` → ``March 20, 2026`` (UTC calendar date)."""
    if not raw or not str(raw).strip():
        return "—"
    s = str(raw).strip()
    try:
        if s.endswith("Z"):
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt = dt.astimezone(timezone.utc)
        return dt.strftime("%B %d, %Y")
    except (ValueError, TypeError, OSError):
        return s
