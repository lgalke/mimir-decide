"""Permissive-license policy (user decision 2026-10-04: permissive only).

Anything unknown, unspecified, non-commercial or otherwise not on the allowlist is excluded.
CC-BY-SA is allowed but flagged in the log: share-alike on model weights is legally unsettled.
Compound license strings ("cc-by-4.0, CC BY 4.0 (DPI)") are allowed only if every part is allowed.
"""
from __future__ import annotations

import re

PERMISSIVE = {
    "apache-2.0", "mit", "bsd", "bsd-2-clause", "bsd-3-clause", "isc",
    "cc0-1.0", "cc-by-4.0", "cc-by-3.0", "cc-by-2.0",
    "cc-by-sa-4.0", "cc-by-sa-3.0", "odc-by", "pddl", "unlicense", "afl-3.0",
}
SHARE_ALIKE = {"cc-by-sa-4.0", "cc-by-sa-3.0"}
NONCOMMERCIAL_RE = re.compile(r"(^|[-_\s])nc([-_\s]|$)|non-?commercial|research[-_ ]only", re.I)
UNKNOWN = {"", "other", "unknown", "unspecified", "none", "null", "n/a"}


def _canon(part: str) -> str:
    s = part.strip().lower()
    s = re.sub(r"\(dpi\)", "", s).strip()
    s = s.replace("_", "-")
    s = re.sub(r"^creative commons\s+", "cc ", s)
    s = re.sub(r"^cc[ -]?by[ -]?sa[ -]?(\d\.\d)", r"cc-by-sa-\1", s)
    s = re.sub(r"^cc[ -]?by[ -]?(\d\.\d)", r"cc-by-\1", s)
    s = re.sub(r"^cc[ -]?0.*$", "cc0-1.0", s)
    s = re.sub(r"^apache([ -]license)?[ ,-]*(version )?2(\.0)?$", "apache-2.0", s)
    s = re.sub(r"^mit([ -]license)?$", "mit", s)
    s = re.sub(r"^bsd[ -]?([23])[ -]?clause.*$", r"bsd-\1-clause", s)
    return s.replace(" ", "-")


def normalise(lic: str | None) -> str:
    if not lic:
        return ""
    return ",".join(_canon(p) for p in lic.split(",") if p.strip())


def check(lic: str | None, license_use: str | None = None) -> tuple[bool, str]:
    """Return (allowed, reason). Reasons: ok, share_alike, unknown_license, non_commercial, not_on_allowlist."""
    parts = [_canon(p) for p in (lic or "").split(",") if p.strip()]
    if not parts or any(p in UNKNOWN for p in parts):
        return False, "unknown_license"
    if any(NONCOMMERCIAL_RE.search(p) for p in parts) or (
        license_use and NONCOMMERCIAL_RE.search(license_use.lower())
    ):
        return False, "non_commercial"
    if not all(p in PERMISSIVE for p in parts):
        return False, "not_on_allowlist"
    return True, "share_alike" if any(p in SHARE_ALIKE for p in parts) else "ok"
