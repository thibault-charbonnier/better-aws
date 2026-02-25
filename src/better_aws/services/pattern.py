import re
from typing import Optional, Sequence

def _has_glob(s: str) -> bool:
    return "*" in s or "?" in s

def _glob_base_dir(pattern: str) -> str:
    m = re.search(r"[\*\?]", pattern)
    base = pattern if not m else pattern[:m.start()]
    if "/" in base:
        return base.rsplit("/", 1)[0] + "/"
    return ""

def _glob_to_regex(pattern: str) -> re.Pattern:
    """
    Convert a minimal glob pattern to a compiled regex.
    Supports:
      *  => any sequence (including '/')
      ?  => any single char (including '/')
    """
    s = re.escape(pattern)
    s = s.replace(r"\*", ".*").replace(r"\?", ".")
    return re.compile(r"^" + s + r"$")

def _norm_ext_set(exts: Optional[Sequence[str]]) -> Optional[set[str]]:
    if exts is None:
        return None
    out: set[str] = set()
    for e in exts:
        if not e:
            continue
        e = e.lower()
        if not e.startswith("."):
            e = "." + e
        out.add(e)
    return out