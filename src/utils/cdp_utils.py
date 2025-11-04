"""
src/utils/cdp_utils.py

Utility functions for CDP web scraping.
"""

import os
import json
import time
import re


def write_jsonl(path, obj):
    """Write object as JSON line to file."""
    with open(path, mode="a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_json_file(path, obj):
    """Write pretty JSON file."""
    with open(path, mode="w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def sanitize_filename(s, default="file"):
    """Sanitize string for use as filename."""
    s = "".join(c for c in s if c.isalnum() or c in ("-", "_", "."))
    return s or default


def build_pair_dir(url: str, ts_ms: int, output_dir: str) -> str:
    """Build per-request directory: date_timestamp_url."""
    date_str = time.strftime("%Y%m%d", time.localtime(ts_ms / 1000))
    url_core = (url or "").split("://", 1)[-1].split("?", 1)[0].strip("/")
    url_core = url_core.replace("/", "_")
    safe_url = sanitize_filename(url_core)[:120] or "url"
    dir_name = f"{date_str}_{ts_ms}_{safe_url}"
    dir_path = os.path.join(output_dir, dir_name)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def get_set_cookie_values(headers):
    """Extract Set-Cookie values from headers."""
    values = []
    if not headers:
        return values
    try:
        for k, v in headers.items():
            if str(k).lower() == "set-cookie":
                if isinstance(v, str):
                    parts = v.split("\n") if "\n" in v else [v]
                    for line in parts:
                        line = line.strip()
                        if line:
                            values.append(line)
                elif isinstance(v, (list, tuple)):
                    for line in v:
                        if line:
                            values.append(str(line))
    except Exception:
        pass
    return values


def cookie_names_from_set_cookie(values):
    """Extract cookie names from Set-Cookie values."""
    names = []
    for sc in values:
        first = sc.split(";", 1)[0]
        name = first.split("=", 1)[0].strip()
        if name:
            names.append(name)
    return names


def blocked_by_regex(url: str, block_regexes: list) -> bool:
    """Check if URL should be blocked by regex patterns."""
    u = url or ""
    for rx in block_regexes:
        if re.search(rx, u, flags=re.IGNORECASE):
            return True
    return False
