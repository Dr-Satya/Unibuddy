"""
Timetable lookup module for UniBuddy chatbot.
Loads timetable.json + sections.json and provides
context-ready text for the LLM prompt.
"""

import json
import os
import re
from typing import List, Dict, Optional, Tuple

_BASE = os.path.join(os.path.dirname(__file__), "..", "UnibuddyTT")

def _load():
    tt_path = os.path.join(_BASE, "timetable.json")
    sec_path = os.path.join(_BASE, "sections.json")
    with open(tt_path, encoding="utf-8") as f:
        timetable: List[Dict] = json.load(f)
    with open(sec_path, encoding="utf-8") as f:
        sections: Dict = json.load(f)
    return timetable, sections

try:
    _TIMETABLE, _SECTIONS = _load()
except Exception as e:
    print(f"[TT] Warning: could not load timetable data: {e}")
    _TIMETABLE, _SECTIONS = [], {}


def get_all_section_keys() -> List[str]:
    return sorted(_SECTIONS.keys())


def get_section_options() -> List[Dict]:
    """Human-readable options for signup dropdown."""
    options = []
    for key in _SECTIONS:
        parts = key.split("|")
        degree = parts[0] if len(parts) > 0 else ""
        year   = parts[1] if len(parts) > 1 else ""
        branch = parts[2] if len(parts) > 2 else ""
        label  = f"{degree} Year {year} - {branch}".replace("_", " ")
        options.append({"label": label, "value": key})
    return sorted(options, key=lambda x: x["label"])


def get_student_timetable(section_key: str) -> Tuple[List[Dict], Optional[str]]:
    """
    Returns (entries, section_key) for a given section key like 'BTECH|2|CSE_B2'.
    Uses the first (most recent) PDF source for that section.
    """
    if not section_key or section_key not in _SECTIONS:
        return [], None

    source_info = _SECTIONS[section_key][0]
    source_file = source_info["file"]
    page        = source_info["page"]

    entries = [
        e for e in _TIMETABLE
        if e.get("source") == source_file and e.get("page") == page
    ]
    return entries, section_key


def format_timetable_for_prompt(entries: List[Dict], section_key: str) -> str:
    """Convert timetable entries to readable text for LLM context."""
    if not entries:
        return "No timetable found for this section."

    days_order = ["Mo", "Tu", "We", "Th", "Fr", "Sa"]
    day_names  = {"Mo": "Monday", "Tu": "Tuesday", "We": "Wednesday",
                  "Th": "Thursday", "Fr": "Friday", "Sa": "Saturday"}

    by_day: Dict[str, List[str]] = {d: [] for d in days_order}

    for e in entries:
        day = e.get("day", "")
        if day in by_day and e.get("course", "FREE") != "FREE":
            line = (
                f"  {e.get('time','')} | {e.get('course','')} "
                f"({e.get('course_code','')}) | {e.get('faculty','')} | Room {e.get('room','')}"
            )
            by_day[day].append(line)

    lines = [f"📅 Timetable for section: {section_key.replace('|', ' | ')}"]
    for d in days_order:
        if by_day[d]:
            lines.append(f"\n{day_names.get(d, d)}:")
            lines.extend(by_day[d])

    if len(lines) == 1:
        return f"No scheduled classes found for section {section_key}."

    return "\n".join(lines)


def is_timetable_query(text: str) -> bool:
    """Quick check if user is asking about timetable/schedule."""
    keywords = [
        "timetable", "time table", "schedule", "class", "lecture",
        "today", "tomorrow", "monday", "tuesday", "wednesday",
        "thursday", "friday", "period", "slot", "room", "subject",
        "kab", "kitne baje", "class hai", "lecture hai", "aaj",
        "kal", "week", "routine"
    ]
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)


def get_timetable_context(section_key: str, user_query: str) -> str:
    """
    Returns timetable context string to inject into LLM prompt.
    Returns empty string if not a timetable query or no section.
    """
    if not section_key:
        return ""
    if not is_timetable_query(user_query):
        return ""
    entries, key = get_student_timetable(section_key)
    return format_timetable_for_prompt(entries, key)
