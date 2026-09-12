"""
Timetable store — multi-turn guided flow to identify the user's class,
then passes the timetable as context to Groq LLM to answer ANY question.

Flow:
  1. Detect timetable intent (or active session)
  2. Ask Year → Branch → Section until class is resolved
  3. Build a compact text representation of the timetable
  4. Send to Groq: "Given this timetable, answer: <user question>"
  5. Return the LLM's focused answer
"""
import json
import os
import re
from datetime import date, datetime
from typing import Optional
from difflib import SequenceMatcher
from collections import deque, OrderedDict
import time

from src.config import settings

# ── Analytics Foundation (Pure Addition - Optional Logging Layer) ────────────────

# In-memory analytics storage (lightweight, no persistence)
_analytics_logs = deque(maxlen=1000)  # Keep only recent 1000 queries

def _get_current_timestamp() -> str:
    """Get current timestamp in ISO format for logging."""
    try:
        return datetime.now().isoformat()
    except Exception:
        return ""

def _get_execution_time_ms(start_time: float) -> float:
    """Calculate execution time in milliseconds from start time."""
    try:
        if start_time <= 0:
            return 0.0
        return (time.time() - start_time) * 1000
    except Exception:
        return 0.0

def _log_query(query: str, handler_type: str, execution_time_ms: float, 
               success: bool = True, error_msg: str = "") -> None:
    """
    Log query execution to in-memory analytics.
    
    Args:
        query: Original user query
        handler_type: Type of handler used ('display', 'count', 'search', 'time_lookup', 'llm_fallback')
        execution_time_ms: Query execution time in milliseconds
        success: Whether query completed successfully
        error_msg: Error message if failed (optional)
    
    Analytics record contains:
        - timestamp: ISO format timestamp
        - query: User query (first 200 chars)
        - handler_type: Which handler processed the query
        - execution_time_ms: Time taken to process
        - success: Success/failure flag
        - error_msg: Error details if applicable
    
    Note: Logging is completely optional and non-blocking.
    If logging fails, execution continues normally without affecting user response.
    """
    try:
        # Truncate query for storage efficiency
        query_truncated = (query[:200] if query else "")
        
        log_entry = {
            "timestamp": _get_current_timestamp(),
            "query": query_truncated,
            "handler_type": handler_type,
            "execution_time_ms": round(execution_time_ms, 2),
            "success": success,
            "error_msg": error_msg[:100] if error_msg else ""
        }
        
        # Add to in-memory deque (will automatically evict old entries at maxlen)
        _analytics_logs.append(log_entry)
        
    except Exception as e:
        # Silently fail - analytics errors should never affect user experience
        pass

def _get_analytics_summary() -> dict:
    """
    Get summary statistics from recent query logs.
    
    Returns:
        {
            "total_queries": int,
            "successful_queries": int,
            "failed_queries": int,
            "success_rate": float (0-100),
            "avg_execution_time_ms": float,
            "handler_breakdown": {
                "display": int,
                "count": int,
                "search": int,
                "time_lookup": int,
                "llm_fallback": int
            },
            "recent_errors": [list of error messages]
        }
    """
    try:
        if not _analytics_logs:
            return {
                "total_queries": 0,
                "successful_queries": 0,
                "failed_queries": 0,
                "success_rate": 0.0,
                "avg_execution_time_ms": 0.0,
                "handler_breakdown": {},
                "recent_errors": []
            }
        
        total = len(_analytics_logs)
        successful = sum(1 for log in _analytics_logs if log.get("success", False))
        failed = total - successful
        
        # Calculate average execution time
        exec_times = [log.get("execution_time_ms", 0) for log in _analytics_logs if log.get("execution_time_ms", 0) > 0]
        avg_exec_time = sum(exec_times) / len(exec_times) if exec_times else 0.0
        
        # Handler breakdown
        handler_breakdown = {}
        for log in _analytics_logs:
            handler = log.get("handler_type", "unknown")
            handler_breakdown[handler] = handler_breakdown.get(handler, 0) + 1
        
        # Recent errors
        recent_errors = [log.get("error_msg") for log in _analytics_logs 
                        if log.get("error_msg") and not log.get("success", False)]
        recent_errors = list(dict.fromkeys(recent_errors))[:5]  # Unique, max 5
        
        return {
            "total_queries": total,
            "successful_queries": successful,
            "failed_queries": failed,
            "success_rate": round((successful / total * 100) if total > 0 else 0, 2),
            "avg_execution_time_ms": round(avg_exec_time, 2),
            "handler_breakdown": handler_breakdown,
            "recent_errors": recent_errors
        }
    except Exception:
        return {}

def _get_recent_logs(limit: int = 20) -> list:
    """
    Get recent query logs for debugging/monitoring.
    
    Args:
        limit: Maximum number of recent logs to return (default 20)
    
    Returns:
        List of recent log entries (newest first)
    """
    try:
        logs_list = list(_analytics_logs)
        return logs_list[-limit:][::-1]  # Return last N logs, newest first
    except Exception:
        return []

def _clear_analytics_logs() -> None:
    """Clear all analytics logs. Useful for testing/debugging."""
    try:
        _analytics_logs.clear()
    except Exception:
        pass


# ── Response Cache Layer (Pure Addition - Lightweight In-Memory Cache) ──────────

# In-memory cache storage (LRU eviction, TTL-based expiration)
_response_cache = OrderedDict()  # {(class_name, normalized_query): cache_entry}
_cache_ttl_seconds = 300  # 5 minutes
_cache_max_size = 200  # Maximum entries before LRU eviction


def _normalize_query(query: str) -> str:
    """Normalize query for cache key matching.
    
    Normalization:
    - lowercase
    - strip whitespace
    - collapse multiple spaces to single space
    - remove punctuation at end
    
    Examples:
        "Show My Timetable" → "show my timetable"
        "Show  My   Timetable" → "show my timetable"
        "Show my timetable?" → "show my timetable"
    """
    try:
        if not query:
            return ""
        
        normalized = query.lower().strip()
        # Collapse multiple spaces
        normalized = re.sub(r'\s+', ' ', normalized)
        # Remove trailing punctuation
        normalized = re.sub(r'[?!.]*$', '', normalized)
        return normalized
    except Exception:
        return query.lower() if query else ""


def _get_cache_key(class_name: str, query: str) -> tuple:
    """Generate cache key from class name and query.
    
    Returns:
        (class_name, normalized_query) tuple
    """
    try:
        normalized = _normalize_query(query)
        return (class_name, normalized)
    except Exception:
        return (class_name, "")


def _get_cached_response(class_name: str, query: str) -> Optional[dict]:
    """
    Retrieve cached response if available and not expired.
    
    Args:
        class_name: Timetable class name
        query: User query
    
    Returns:
        {
            "response_html": str,
            "handler_type": str,
            "created_at": str (ISO format),
            "age_seconds": int
        }
        or None if not found or expired
    
    Cache lookup is completely optional and non-blocking.
    If cache fails, returns None and continues normally.
    """
    try:
        key = _get_cache_key(class_name, query)
        
        if key not in _response_cache:
            return None
        
        cache_entry = _response_cache[key]
        created_at = cache_entry.get("created_at", 0)
        response_html = cache_entry.get("response_html", "")
        handler_type = cache_entry.get("handler_type", "")
        
        # Check TTL (5 minutes = 300 seconds)
        age = time.time() - created_at
        if age > _cache_ttl_seconds:
            # Expired - remove and return None
            del _response_cache[key]
            return None
        
        # Move to end (mark as recently used for LRU)
        _response_cache.move_to_end(key)
        
        return {
            "response_html": response_html,
            "handler_type": handler_type,
            "created_at": cache_entry.get("created_at_iso", ""),
            "age_seconds": int(age)
        }
    
    except Exception:
        # Cache errors never affect functionality
        return None


def _store_cached_response(class_name: str, query: str, response_html: str, 
                          handler_type: str = "") -> None:
    """
    Store response in cache with TTL and LRU eviction.
    
    Args:
        class_name: Timetable class name
        query: User query
        response_html: HTML response to cache
        handler_type: Type of handler used ('display', 'count', 'search', etc.)
    
    Features:
    - TTL: 5 minutes (300 seconds)
    - LRU eviction: Remove oldest when cache full (max 200 entries)
    - Non-blocking: Cache errors never affect functionality
    """
    try:
        if not response_html or len(response_html) == 0:
            return  # Don't cache empty responses
        
        key = _get_cache_key(class_name, query)
        current_time = time.time()
        
        # Create cache entry
        cache_entry = {
            "response_html": response_html,
            "handler_type": handler_type,
            "created_at": current_time,
            "created_at_iso": datetime.now().isoformat()
        }
        
        # LRU eviction: if cache is full, remove oldest entry
        if len(_response_cache) >= _cache_max_size and key not in _response_cache:
            # Remove oldest entry (first item in OrderedDict)
            _response_cache.popitem(last=False)
        
        # Add or update cache entry (move to end)
        _response_cache[key] = cache_entry
        _response_cache.move_to_end(key)
    
    except Exception:
        # Cache errors never affect functionality
        pass


def _clear_response_cache() -> None:
    """Clear all cached responses. Useful for testing/debugging."""
    try:
        _response_cache.clear()
    except Exception:
        pass


def _get_cache_stats() -> dict:
    """
    Get cache statistics for monitoring/debugging.
    
    Returns:
        {
            "cache_size": int,
            "max_size": int,
            "utilization_percent": float,
            "entries": [list of cache entries with metadata],
            "ttl_seconds": int,
            "expired_entries": int
        }
    """
    try:
        total_entries = len(_response_cache)
        
        # Count expired entries
        current_time = time.time()
        expired_count = 0
        entries_info = []
        
        for key, entry in list(_response_cache.items()):
            created_at = entry.get("created_at", 0)
            age = current_time - created_at
            
            if age > _cache_ttl_seconds:
                expired_count += 1
            
            class_name, query = key
            entries_info.append({
                "class_name": class_name[:50],  # Truncate for readability
                "query": query[:80],            # Truncate for readability
                "handler_type": entry.get("handler_type", ""),
                "age_seconds": int(age),
                "expired": age > _cache_ttl_seconds
            })
        
        return {
            "cache_size": total_entries,
            "max_size": _cache_max_size,
            "utilization_percent": round((total_entries / _cache_max_size * 100) if _cache_max_size > 0 else 0, 2),
            "entries": entries_info,
            "ttl_seconds": _cache_ttl_seconds,
            "expired_entries": expired_count
        }
    
    except Exception:
        return {
            "cache_size": 0,
            "max_size": _cache_max_size,
            "utilization_percent": 0,
            "entries": [],
            "ttl_seconds": _cache_ttl_seconds,
            "expired_entries": 0
        }


def _evict_expired_cache_entries() -> int:
    """
    Remove all expired cache entries.
    
    Returns:
        Number of entries evicted
    
    Useful for maintenance/cleanup.
    """
    try:
        current_time = time.time()
        evicted = 0
        
        keys_to_remove = []
        for key, entry in _response_cache.items():
            created_at = entry.get("created_at", 0)
            age = current_time - created_at
            if age > _cache_ttl_seconds:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del _response_cache[key]
            evicted += 1
        
        return evicted
    
    except Exception:
        return 0


# ── Timetable Upload Validation (Admin Only - Pure Addition) ───────────────────

def _validate_day(day: str) -> bool:
    """Validate if day is in allowed set."""
    allowed_days = {"Mo", "Tu", "We", "Th", "Fr", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"}
    return day in allowed_days if day else False


def _validate_time_format(time_str: str) -> bool:
    """
    Validate time format.
    
    Allowed formats:
    - "HH:MM-HH:MM" (e.g., "9:10-10:00")
    - "H:MM-HH:MM" (e.g., "9:10-10:00")
    
    Also validates that minutes are 0-59.
    
    Returns True if valid, False otherwise.
    """
    if not time_str or not isinstance(time_str, str):
        return False
    
    # Check format: should be "HH:MM-HH:MM" or similar
    pattern = r'^(\d{1,2}):(\d{2})-(\d{1,2}):(\d{2})$'
    match = re.match(pattern, time_str.strip())
    
    if not match:
        return False
    
    # Extract minutes and validate they are 0-59
    start_min = int(match.group(2))
    end_min = int(match.group(4))
    
    if start_min > 59 or end_min > 59:
        return False
    
    return True


def _validate_timetable_row(row: dict, row_index: int) -> list:
    """
    Validate a single timetable row.
    
    Args:
        row: Dictionary containing timetable entry
        row_index: Row number (for error messages)
    
    Returns:
        List of error messages (empty if valid)
    
    Checks:
    - Required fields present
    - No empty values
    - Valid day
    - Valid time format
    """
    errors = []
    
    # Check required fields exist
    required_fields = ["day", "time", "course", "faculty", "room"]
    for field in required_fields:
        if field not in row:
            errors.append(f"Row {row_index}: Missing required field '{field}'")
    
    if errors:  # Skip further checks if fields missing
        return errors
    
    # Check for empty/whitespace-only values
    day = str(row.get("day", "")).strip()
    time_val = str(row.get("time", "")).strip()
    course = str(row.get("course", "")).strip()
    faculty = str(row.get("faculty", "")).strip()
    room = str(row.get("room", "")).strip()
    
    if not day:
        errors.append(f"Row {row_index}: Empty day field")
    if not time_val:
        errors.append(f"Row {row_index}: Empty time field")
    if not course:
        errors.append(f"Row {row_index}: Empty course field")
    if not faculty:
        errors.append(f"Row {row_index}: Empty faculty field")
    if not room:
        errors.append(f"Row {row_index}: Empty room field")
    
    # Skip further validation if critical fields empty
    if not day or not time_val or not course:
        return errors
    
    # Validate day
    if not _validate_day(day):
        errors.append(f"Row {row_index}: Invalid day '{day}' (allowed: Mo, Tu, We, Th, Fr)")
    
    # Validate time format
    if not _validate_time_format(time_val):
        errors.append(f"Row {row_index}: Invalid time format '{time_val}' (expected: HH:MM-HH:MM)")
    
    # Check for FREE placeholder (allowed)
    if course.upper() == "FREE" or faculty.upper() == "FREE" or room.upper() == "FREE":
        pass  # FREE slots are allowed
    
    return errors


def _validate_timetable_json(timetable: list) -> dict:
    """
    Comprehensive validation of timetable JSON before saving.
    
    Args:
        timetable: List of timetable rows
    
    Returns:
        {
            "valid": bool,
            "errors": [list of error messages],
            "warnings": [list of warning messages],
            "statistics": {
                "total_rows": int,
                "non_free_rows": int,
                "unique_courses": int,
                "unique_faculty": int,
                "unique_rooms": int,
                "unique_days": int,
                "courses": [list],
                "faculty": [list],
                "rooms": [list],
                "days": [list]
            }
        }
    
    Validation checks:
    1. Timetable is not empty
    2. Each row has required fields
    3. No empty values in required fields
    4. Valid day values
    5. Valid time format
    6. No duplicate entries (same day, time, course)
    7. Collects statistics
    """
    errors = []
    warnings = []
    
    # Check if timetable is empty
    if not timetable or len(timetable) == 0:
        return {
            "valid": False,
            "errors": ["Timetable is empty (0 rows)"],
            "warnings": [],
            "statistics": {
                "total_rows": 0,
                "non_free_rows": 0,
                "unique_courses": 0,
                "unique_faculty": 0,
                "unique_rooms": 0,
                "unique_days": 0,
                "courses": [],
                "faculty": [],
                "rooms": [],
                "days": []
            }
        }
    
    # Validate each row
    duplicate_entries = set()
    seen_entries = {}
    
    courses_set = set()
    faculty_set = set()
    rooms_set = set()
    days_set = set()
    non_free_rows = 0
    
    for idx, row in enumerate(timetable, 1):
        if not isinstance(row, dict):
            errors.append(f"Row {idx}: Not a dictionary")
            continue
        
        # Validate row structure
        row_errors = _validate_timetable_row(row, idx)
        errors.extend(row_errors)
        
        # Collect statistics from valid rows
        day = str(row.get("day", "")).strip()
        time_val = str(row.get("time", "")).strip()
        course = str(row.get("course", "")).strip()
        faculty = str(row.get("faculty", "")).strip()
        room = str(row.get("room", "")).strip()
        
        if day and time_val and course:
            days_set.add(day)
            
            # Track non-FREE entries
            if course.upper() != "FREE":
                non_free_rows += 1
                courses_set.add(course)
            
            if faculty and faculty.upper() != "FREE":
                faculty_set.add(faculty)
            if room and room.upper() != "FREE":
                rooms_set.add(room)
            
            # Detect duplicates (same day, time, course)
            entry_key = (day, time_val, course)
            if entry_key in seen_entries:
                if entry_key not in duplicate_entries:
                    duplicate_entries.add(entry_key)
                    dup_row = seen_entries[entry_key]
                    errors.append(f"Duplicate entry: {day} {time_val} {course} (rows {dup_row} and {idx})")
            else:
                seen_entries[entry_key] = idx
    
    # Warnings (not blocking)
    
    # Check for duplicate faculty names with different spacing
    faculty_list = list(faculty_set)
    for i, f1 in enumerate(faculty_list):
        for f2 in faculty_list[i+1:]:
            if f1.lower().replace(" ", "") == f2.lower().replace(" ", ""):
                warnings.append(f"Faculty name formatting inconsistency: '{f1}' vs '{f2}'")
    
    # Check for very short subject names (potential OCR errors)
    for course in courses_set:
        if len(course) <= 2:
            warnings.append(f"Very short course name: '{course}' (possible OCR error)")
    
    # Collect statistics
    statistics = {
        "total_rows": len(timetable),
        "non_free_rows": non_free_rows,
        "unique_courses": len(courses_set),
        "unique_faculty": len(faculty_set),
        "unique_rooms": len(rooms_set),
        "unique_days": len(days_set),
        "courses": sorted(list(courses_set)),
        "faculty": sorted(list(faculty_set)),
        "rooms": sorted(list(rooms_set)),
        "days": sorted(list(days_set))
    }
    
    # Determine validity
    valid = len(errors) == 0
    
    return {
        "valid": valid,
        "errors": errors,
        "warnings": warnings,
        "statistics": statistics
    }

_HERE = os.path.dirname(os.path.abspath(__file__))
STORE_PATH   = os.path.normpath(os.path.join(_HERE, '..', 'data', 'timetables.json'))
SESSION_PATH = os.path.normpath(os.path.join(_HERE, '..', 'data', 'timetable_sessions.json'))

WEEKDAY_TO_ABBR = ["Mo", "Tu", "We", "Th", "Fr"]
DAY_FULL = {"Mo": "Monday", "Tu": "Tuesday", "We": "Wednesday", "Th": "Thursday", "Fr": "Friday"}

TIMETABLE_INTENTS = [
    'class', 'classroom', 'room', 'lecture', 'timetable', 'schedule',
    'subject', 'slot', 'period', 'today', 'tomorrow',
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
    'free', 'gap', 'break', 'next class', 'when is', 'where is',
    'what do i have', 'my schedule', 'my class', 'which subject',
    'who teaches', 'how many class', 'how many period', 'how many lab',
    'first class', 'last class',
]

# Queries that should NEVER be handled by timetable, even inside an active session
TIMETABLE_ESCAPE = [
    'faculty info', 'about faculty', 'tell me about', 'who is', 'details of',
    'profile of', 'mentor', 'mentee', 'fee', 'fees', 'admission',
    'professor', 'dr.', 'sir teach', 'sir profile',
]

# ── persistence ────────────────────────────────────────────────────────────────

def _load() -> dict:
    if os.path.exists(STORE_PATH):
        try:
            with open(STORE_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save(data: dict):
    os.makedirs(os.path.dirname(STORE_PATH), exist_ok=True)
    with open(STORE_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[Timetable] Written to {STORE_PATH}")


def save_timetable(parsed_result: dict):
    store = _load()
    for tt in parsed_result.get('timetables', []):
        class_name = (tt.get('class') or '').strip()
        if not class_name:
            class_name = f"Page {tt.get('page', '?')}"
        store[class_name] = tt
    _save(store)
    print(f"[Timetable] Saved {len(store)} class(es) to {STORE_PATH}")
    return list(store.keys())


def get_all_classes() -> list:
    return list(_load().keys())


# ── session state ──────────────────────────────────────────────────────────────

def _load_sessions() -> dict:
    if os.path.exists(SESSION_PATH):
        try:
            with open(SESSION_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_sessions(s: dict):
    os.makedirs(os.path.dirname(SESSION_PATH), exist_ok=True)
    with open(SESSION_PATH, 'w', encoding='utf-8') as f:
        json.dump(s, f, ensure_ascii=False, indent=2)


def _get_session(sid: str) -> dict:
    return _load_sessions().get(sid, {})


def _set_session(sid: str, data: dict):
    s = _load_sessions()
    s[sid] = data
    _save_sessions(s)


def _clear_session(sid: str):
    s = _load_sessions()
    s.pop(sid, None)
    _save_sessions(s)


# ── JSON Lookup Engine (reusable helpers) ──────────────────────────────────────
# Pure functions for querying timetable data. No LLM, no side effects.
# Used internally by answer generation logic.

def _is_free_slot(item: dict) -> bool:
    """Check if a schedule item is a free slot."""
    if not item:
        return False
    course = (item.get('course') or '').strip().upper()
    return course in ('FREE', '—', 'FREE PERIOD', 'BREAK')


def get_classes_for_day(schedule: list, day: str) -> list:
    """Get all class items for a specific day (deduplicated)."""
    if not schedule or not day:
        return []
    
    day_upper = day.upper()
    classes = []
    
    # New format: schedule is list of slots with {'slot': N, 'time': '...', 'classes': {'Mo': '...', 'Tu': '...', ...}}
    # Old format: schedule is list of items with {'day': 'Mo', 'time': '...', 'course': '...', ...}
    
    for item in schedule:
        # Check if this is the new slot-based format
        if 'classes' in item and isinstance(item['classes'], dict):
            # New format: extract class string for this day
            class_str = item['classes'].get(day, '') or item['classes'].get(day_upper, '')
            if class_str and class_str.strip().upper() not in ('FREE', '—', 'FREE PERIOD', 'BREAK'):
                # Parse the class string into a normalized item dict
                time_slot = item.get('time', '')
                parsed_item = {
                    'time': time_slot,
                    'day': day,
                    'course': class_str.strip(),
                    'faculty': '',
                    'room': '',
                    'raw': class_str
                }
                classes.append(parsed_item)
        else:
            # Old format: check item['day'] directly
            if (item.get('day') or '').upper() == day_upper and not _is_free_slot(item):
                classes.append(item)
    
    # Deduplicate by (time, course) composite key
    seen = set()
    unique_classes = []
    for item in classes:
        key = (
            (item.get('time') or '').upper(),
            (item.get('course') or '').upper(),
        )
        if key not in seen:
            seen.add(key)
            unique_classes.append(item)
    
    return unique_classes


def get_free_slots_for_day(schedule: list, day: str) -> list:
    """Get all free slots for a specific day."""
    if not schedule or not day:
        return []
    day_upper = day.upper()
    return [item for item in schedule if (item.get('day') or '').upper() == day_upper and _is_free_slot(item)]


def get_classes_for_time(schedule: list, time_slot: str) -> list:
    """Get all class items for a specific time slot."""
    if not schedule or not time_slot:
        return []
    time_normalized = time_slot.lower().strip()
    return [item for item in schedule if (item.get('time') or '').lower() == time_normalized and not _is_free_slot(item)]


def get_subject(schedule: list, day: str = None, time_slot: str = None) -> str:
    """Get subject/course name for a specific day/time combination."""
    if not schedule:
        return None
    
    filters = []
    if day:
        filters.append(lambda item: (item.get('day') or '').upper() == day.upper())
    if time_slot:
        filters.append(lambda item: (item.get('time') or '').lower() == time_slot.lower())
    
    for item in schedule:
        if all(f(item) for f in filters) and not _is_free_slot(item):
            return (item.get('course') or '').strip()
    return None


def get_faculty(schedule: list, day: str = None, time_slot: str = None) -> str:
    """Get faculty name for a specific day/time combination."""
    if not schedule:
        return None
    
    filters = []
    if day:
        filters.append(lambda item: (item.get('day') or '').upper() == day.upper())
    if time_slot:
        filters.append(lambda item: (item.get('time') or '').lower() == time_slot.lower())
    
    for item in schedule:
        if all(f(item) for f in filters) and not _is_free_slot(item):
            return (item.get('faculty') or '').strip()
    return None


def get_room(schedule: list, day: str = None, time_slot: str = None) -> str:
    """Get room/location for a specific day/time combination."""
    if not schedule:
        return None
    
    filters = []
    if day:
        filters.append(lambda item: (item.get('day') or '').upper() == day.upper())
    if time_slot:
        filters.append(lambda item: (item.get('time') or '').lower() == time_slot.lower())
    
    for item in schedule:
        if all(f(item) for f in filters) and not _is_free_slot(item):
            return (item.get('room') or '').strip()
    return None


def get_first_class(schedule: list, day: str = None) -> dict:
    """Get the first class of the day (by time order, not index)."""
    if not schedule:
        return None
    
    if day:
        classes = get_classes_for_day(schedule, day)
    else:
        classes = [item for item in schedule if not _is_free_slot(item)]
    
    # Sort by time to ensure chronological order
    if classes:
        try:
            classes_sorted = sorted(classes, key=lambda x: x.get('time', ''))
            return classes_sorted[0]
        except Exception:
            return classes[0]
    
    return None


def get_last_class(schedule: list, day: str = None) -> dict:
    """Get the last class of the day (by time order, not index)."""
    if not schedule:
        return None
    
    if day:
        classes = get_classes_for_day(schedule, day)
    else:
        classes = [item for item in schedule if not _is_free_slot(item)]
    
    # Sort by time to ensure chronological order
    if classes:
        try:
            classes_sorted = sorted(classes, key=lambda x: x.get('time', ''))
            return classes_sorted[-1]
        except Exception:
            return classes[-1]
    
    return None


def count_classes(schedule: list, day: str = None) -> int:
    """Count total classes."""
    if not schedule:
        return 0
    
    if day:
        return len(get_classes_for_day(schedule, day))
    else:
        return len([item for item in schedule if not _is_free_slot(item)])


def count_free_slots(schedule: list, day: str = None) -> int:
    """Count total free slots."""
    if not schedule:
        return 0
    
    if day:
        return len(get_free_slots_for_day(schedule, day))
    else:
        return len([item for item in schedule if _is_free_slot(item)])


def get_all_subjects(schedule: list) -> list:
    """Get all unique subject names (excluding free slots)."""
    if not schedule:
        return []
    subjects = set()
    for item in schedule:
        if not _is_free_slot(item):
            subject = (item.get('course') or '').strip()
            if subject:
                subjects.add(subject)
    return sorted(list(subjects))


def _build_subject_alias_map(schedule: list) -> dict:
    """
    Build a lightweight alias map for subject lookups.
    Maps common abbreviations/variations to course codes.
    
    Derived from actual timetable data (no hardcoded knowledge).
    Examples: "DBMS" → "CSE3013", "WAD" → "CSE3013", etc.
    
    Returns: {"alias": "course_code", ...}
    """
    if not schedule:
        return {}
    
    alias_map = {}
    
    for item in schedule:
        if _is_free_slot(item):
            continue
        
        course_full = (item.get('course') or '').strip()
        if not course_full:
            continue
        
        # Parse course code and title
        # Typical format: "CSE3013 Database Management Systems"
        parts = course_full.split(None, 1)  # Split on first whitespace
        if len(parts) == 2:
            course_code = parts[0]
            course_title = parts[1]
            
            # Generate aliases from title
            # Split title into words and use meaningful ones as aliases
            title_words = course_title.split()
            
            # Add full course code as alias
            alias_map[course_code] = course_code
            
            # Add title as alias
            alias_map[course_title.lower()] = course_code
            
            # Add abbreviations (first letter of each word)
            abbrev = ''.join(w[0].upper() for w in title_words if w)
            if abbrev and len(abbrev) >= 2:  # Only if meaningful
                alias_map[abbrev.lower()] = course_code
                alias_map[abbrev] = course_code
            
            # Add individual significant words (3+ chars) as aliases
            for word in title_words:
                if len(word) >= 3:
                    alias_map[word.lower()] = course_code
        else:
            # No title, just code
            course_code = parts[0] if parts else course_full
            alias_map[course_code] = course_code
    
    return alias_map


def _resolve_subject_via_alias(query: str, schedule: list) -> Optional[str]:
    """
    Resolve subject query using alias map.
    Returns the course code if found, None otherwise.
    """
    if not query or not schedule:
        return None
    
    alias_map = _build_subject_alias_map(schedule)
    if not alias_map:
        return None
    
    query_lower = query.lower()
    
    # Direct alias match
    if query_lower in alias_map:
        return alias_map[query_lower]
    
    # Fuzzy match on aliases
    aliases = list(alias_map.keys())
    match, confidence, _ = _fuzzy_match(query, aliases, threshold=0.6)
    
    if match and confidence >= 0.65:
        return alias_map.get(match)
    
    return None


def get_all_faculty(schedule: list) -> list:
    """Get all unique faculty names (excluding free slots)."""
    if not schedule:
        return []
    faculty_list = set()
    for item in schedule:
        if not _is_free_slot(item):
            faculty = (item.get('faculty') or '').strip()
            if faculty:
                faculty_list.add(faculty)
    return sorted(list(faculty_list))


def get_all_rooms(schedule: list) -> list:
    """Get all unique room numbers/codes (excluding free slots)."""
    if not schedule:
        return []
    rooms = set()
    for item in schedule:
        if not _is_free_slot(item):
            room = (item.get('room') or '').strip()
            if room:
                rooms.add(room)
    return sorted(list(rooms))


def find_subject(schedule: list, query: str) -> list:
    """Find all items matching a subject/course name (case-insensitive, deduplicated)."""
    if not schedule or not query:
        return []
    query_lower = query.lower()
    matches = [item for item in schedule if query_lower in (item.get('course') or '').lower() and not _is_free_slot(item)]
    
    # Deduplicate by composite key
    seen = set()
    unique_matches = []
    for item in matches:
        key = (
            (item.get('day') or '').upper(),
            (item.get('time') or '').upper(),
            (item.get('course') or '').upper()
        )
        if key not in seen:
            seen.add(key)
            unique_matches.append(item)
    
    return unique_matches


def find_faculty(schedule: list, query: str) -> list:
    """Find all items matching a faculty name (case-insensitive, deduplicated)."""
    if not schedule or not query:
        return []
    query_lower = query.lower()
    matches = [item for item in schedule if query_lower in (item.get('faculty') or '').lower() and not _is_free_slot(item)]
    
    # Deduplicate by composite key
    seen = set()
    unique_matches = []
    for item in matches:
        key = (
            (item.get('day') or '').upper(),
            (item.get('time') or '').upper(),
            (item.get('faculty') or '').upper()
        )
        if key not in seen:
            seen.add(key)
            unique_matches.append(item)
    
    return unique_matches


def find_room(schedule: list, query: str) -> list:
    """Find all items matching a room code (case-insensitive, deduplicated)."""
    if not schedule or not query:
        return []
    query_lower = query.lower()
    matches = [item for item in schedule if query_lower in (item.get('room') or '').lower() and not _is_free_slot(item)]
    
    # Deduplicate by composite key
    seen = set()
    unique_matches = []
    for item in matches:
        key = (
            (item.get('day') or '').upper(),
            (item.get('time') or '').upper(),
            (item.get('room') or '').upper()
        )
        if key not in seen:
            seen.add(key)
            unique_matches.append(item)
    
    return unique_matches


def get_schedule_by_day(schedule: list) -> dict:
    """Organize schedule by day. Returns {day: [items]}."""
    if not schedule:
        return {}
    by_day = {}
    for item in schedule:
        day = (item.get('day') or 'UNKNOWN').strip()
        if day not in by_day:
            by_day[day] = []
        by_day[day].append(item)
    return by_day


def get_schedule_by_time(schedule: list) -> dict:
    """Organize schedule by time slot. Returns {time: [items]}."""
    if not schedule:
        return {}
    by_time = {}
    for item in schedule:
        time = (item.get('time') or 'UNKNOWN').strip()
        if time not in by_time:
            by_time[time] = []
        by_time[time].append(item)
    return by_time


# ── timetable → compact text ───────────────────────────────────────────────────

# ── LLM answerer ───────────────────────────────────────────────────────────────

def _ask_llm(timetable_text: str, user_question: str, history: list = None) -> str:
    """Send timetable context + user question to Groq and return focused answer."""
    try:
        import os
        from pathlib import Path
        from dotenv import load_dotenv
        from groq import Groq

        env_path = Path(_HERE).parent / '.env'
        load_dotenv(dotenv_path=env_path, override=True)

        api_key = os.environ.get('GROQ_API_KEY', '').strip()
        if not api_key or api_key == "your_groq_api_key_here":
            print(f"[Timetable LLM] No GROQ_API_KEY — env path checked: {env_path}")
            return None

        client = Groq(api_key=api_key)

        system = (
            "You are a university timetable assistant. "
            "Answer the student's question using ONLY the timetable data provided.\n"
            "CRITICAL RULES:\n"
            "- Each column in the timetable is a SEPARATE day. NEVER mix data between columns.\n"
            "- Monday column data is ONLY for Monday. Tuesday column is ONLY for Tuesday. etc.\n"
            "- A cell showing '—' means NO CLASS for that day/slot. Do not invent classes.\n"
            "- Use the VERIFIED CLASS COUNT PER DAY section for counts — do not recount.\n"
            "- 'free periods/slots' = slots where the day column shows '—'.\n"
            "- 'which faculty/teacher' = give teacher name for that slot/day.\n"
            "- 'where is my class' = give room code (e.g. B116) and subject name.\n"
            "- 'after X pm' = only slots starting at or after that time.\n"
            "- 'tomorrow' = use the Tomorrow day shown in the timetable header.\n"
            "- 'today' = use the Today day shown in the timetable header.\n"
            "- When listing classes for a day, ONLY list entries from that day's column.\n"
            "- Be concise. 1-5 lines max unless listing multiple items.\n"
            "- Do NOT include URLs, faculty profile links, or external sources.\n"
            "- If not found in timetable, say 'No class found for that time/day'."
        )

        messages = [{"role": "system", "content": system}]

        # Add recent conversation history so "name them" has context
        if history:
            for h in history[-4:]:
                if h.get('user'):
                    messages.append({"role": "user", "content": h['user']})
                if h.get('assistant'):
                    clean = re.sub(r'<[^>]+>', ' ', h['assistant']).strip()
                    messages.append({"role": "assistant", "content": clean})

        messages.append({
            "role": "user",
            "content": f"TIMETABLE DATA:\n{timetable_text}\n\nSTUDENT QUESTION: {user_question}"
        })

        print(f"[Timetable LLM] Calling Groq for: '{user_question[:80]}'")
        resp = client.chat.completions.create(
            messages=messages,
            model=getattr(settings, 'GROQ_MODEL', 'openai/gpt-oss-20b'),
            max_tokens=400,
            temperature=0.0,
        )
        answer = resp.choices[0].message.content.strip()
        print(f"[Timetable LLM] Answer: {answer[:120]}")

        answer = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', answer)
        answer = re.sub(r'\*(.+?)\*', r'<em>\1</em>', answer)
        answer = answer.replace('\n', '<br>')
        return f"<div>{answer}</div>"

    except Exception as e:
        print(f"[Timetable LLM] Exception: {type(e).__name__}: {e}")
        return None


# ── class name parsing / filtering ────────────────────────────────────────────

def _parse_class_name(name: str) -> dict:
    n = name.lower()
    year = next((y for y in ['1st', '2nd', '3rd', '4th', '5th'] if y in n), None)
    degree = next((d for d in ['b.tech', 'bca', 'mca', 'm.tech', 'b.sc', 'm.sc', 'diploma', 'bba', 'mba'] if d in n), None)
    branch = next((b for b in ['cse', 'ece', 'me', 'ce', 'bme', 'data science', 'fire and safety',
                                'aerospace', 'biotechnology', 'microbiology', 'f.sc', 'aiml'] if b in n), None)
    sec_m = re.search(r'\(([^)]+)\)', name)
    section = sec_m.group(1).strip() if sec_m else None
    if not section:
        m2 = re.search(r'\b([A-Z][A-Z0-9]?|Core|DA|ML|SI)\b\.?$', name.strip())
        section = m2.group(1) if m2 else None
    return {'year': year, 'degree': degree, 'branch': branch, 'section': section}


def _filter_classes(year=None, degree=None, branch=None, section=None) -> list:
    store = _load()
    results = []
    for name in store:
        p = _parse_class_name(name)
        if year and p['year'] and year.lower() not in p['year']:
            continue
        if degree and p['degree'] and degree.lower() not in (p['degree'] or ''):
            continue
        # branch can match either the branch field OR the degree field
        # e.g. user says "diploma" → matches degree='diploma'
        # user says "cse" → matches branch='cse'
        if branch:
            branch_l = branch.lower()
            matches_branch = p['branch'] and branch_l in (p['branch'] or '')
            matches_degree = p['degree'] and branch_l in (p['degree'] or '')
            if not matches_branch and not matches_degree:
                continue
        if section and p['section'] and section.lower() not in (p['section'] or '').lower():
            continue
        results.append(name)
    return results


def _unique_branches(names: list) -> list:
    seen, out = set(), []
    for name in names:
        p = _parse_class_name(name)
        # Build a readable label
        degree = p.get('degree') or ''
        branch = p.get('branch') or ''
        key = f"{degree} {branch}".strip()
        if not key or key == ' ':
            continue  # skip entries with no identifiable branch
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _unique_sections(names: list) -> list:
    """Return clean section labels for display."""
    seen, out = set(), []
    for name in names:
        p = _parse_class_name(name)
        sec = p.get('section')
        # Use section code if available, else strip year/degree for cleaner label
        if sec:
            label = sec
        else:
            label = re.sub(r'^(1st|2nd|3rd|4th|5th)\s+year\s+', '', name.strip().rstrip('.'), flags=re.I).strip()
        if label not in seen:
            seen.add(label)
            out.append(label)
    return out


def _match_branch_from_reply(msg: str, year: str = None) -> Optional[str]:
    # Keyword-based branch matching (exact match priority)
    branch_keywords = [
        ('diploma cse', 'diploma'), ('diploma', 'diploma'),
        ('cse', 'cse'), ('ece', 'ece'), ('bme', 'bme'), ('data science', 'data science'),
        ('fire', 'fire and safety'), ('aerospace', 'aerospace'),
        ('biotechnology', 'biotechnology'), ('microbiology', 'microbiology'),
        ('aiml', 'aiml'), ('bca', 'bca'), ('mca', 'mca'),
        ('b.tech', 'b.tech'), ('btech', 'b.tech'), ('b.sc', 'b.sc'), ('bsc', 'b.sc'),
        ('m.sc', 'm.sc'), ('msc', 'm.sc'), ('m.tech', 'm.tech'),
        ('computer science', 'cse'), ('me ', 'me'), ('ce ', 'ce'),
    ]
    
    for kw, mapped in branch_keywords:
        # Multi-word keywords or keywords with trailing space: use substring
        if len(kw.split()) > 1 or kw.endswith(' '):
            if kw in msg:
                return mapped
        else:
            # Single-word keywords: use word boundary to avoid false positives
            pattern = r'\b' + re.escape(kw.rstrip()) + r'\b'
            if re.search(pattern, msg, re.IGNORECASE):
                return mapped
    
    # Fallback: match against available branches for year
    store = _load()
    candidates = _filter_classes(year=year) if year else list(store.keys())
    for b in _unique_branches(candidates):
        if any(w in msg.lower() for w in b.lower().split() if len(w) > 1):
            return b.split()[-1]
    return None


def _match_section_from_reply(msg: str, candidates: list) -> Optional[str]:
    """Match user's section reply to a class name."""
    msg_up = msg.upper().strip()
    msg_lower = msg.lower().strip()

    # Exact section match (e.g. "AIML A", "DA", "ML", "B1", "Core")
    for name in candidates:
        p = _parse_class_name(name)
        sec = (p.get('section') or '').upper()
        if sec and (sec == msg_up or sec in msg_up):
            return name

    # Match significant words from class name (e.g. "aiml a", "diploma cse")
    for name in candidates:
        words = [w for w in re.findall(r'\w+', name.lower())
                 if len(w) > 2 and w not in ('year', 'the', 'and', 'sec', 'section', 'btech', 'bca')]
        if sum(1 for w in words if w in msg_lower) >= 2:
            return name

    # Single candidate
    if len(candidates) == 1:
        return candidates[0]
    return None


def _extract_inline_filters(msg: str) -> dict:
    m = msg.lower()
    filters = {}
    
    # Year extraction with word boundary (avoid matching 'third' in 'third-party', etc.)
    for y in ['1st', '2nd', '3rd', '4th', '5th', 'first', 'second', 'third', 'fourth']:
        pattern = r'\b' + re.escape(y) + r'\b'
        if re.search(pattern, m):
            filters['year'] = {'first': '1st', 'second': '2nd', 'third': '3rd', 'fourth': '4th'}.get(y, y)
            break
    
    # Degree/Program extraction with word boundary (avoid 'diploma' in 'diplomacy', 'bca' in 'subclass')
    for d in ['b.tech', 'btech', 'bca', 'mca', 'm.tech', 'b.sc', 'bsc', 'm.sc', 'diploma']:
        pattern = r'\b' + re.escape(d) + r'\b'
        if re.search(pattern, m):
            filters['degree'] = d.replace('btech', 'b.tech').replace('bsc', 'b.sc').replace('msc', 'm.sc').replace('mtech', 'm.tech')
            break
    
    # Branch extraction with word boundary (critical fix: avoid 'me' in 'monday')
    for b in ['cse', 'ece', 'me', 'bme', 'ce', 'data science', 'fire', 'aerospace',
              'biotechnology', 'microbiology', 'aiml']:
        # Multi-word branches: use substring match (specific enough)
        if len(b.split()) > 1:
            if b in m:
                filters['branch'] = b
                break
        else:
            # Single-word branches: use word boundary match
            pattern = r'\b' + re.escape(b) + r'\b'
            if re.search(pattern, m):
                filters['branch'] = b
                break
    
    # Section extraction (already uses word boundary via regex)
    sec = re.search(r'\b(sec(?:tion)?\s*([a-z]\d?)|([a-z]\d)|da|ml|core|si)\b', m)
    if sec:
        filters['section'] = (sec.group(2) or sec.group(3) or sec.group(1)).upper()
    return filters


# ── query type classifier (internal) ───────────────────────────────────────────


# ── intent detection ───────────────────────────────────────────────────────────

def is_timetable_intent(msg: str) -> bool:
    m = msg.lower()
    return any(k in m for k in TIMETABLE_INTENTS)


def is_timetable_escape(msg: str) -> bool:
    """Returns True if this message should bypass timetable even in an active session."""
    m = msg.lower()
    return any(k in m for k in TIMETABLE_ESCAPE)


def has_active_timetable_session(session_id: str) -> bool:
    """Returns True if this session is mid-flow in the timetable guided conversation."""
    if not session_id:
        return False
    state = _get_session(session_id)
    return bool(state)


# ── main entry point ───────────────────────────────────────────────────────────

def answer_timetable_query(user_message: str, session_id: str = None, history: list = None) -> Optional[str]:
    """
    Returns HTML answer string, or None to fall through to general RAG.
    """
    # Generate session_id if not provided (ensures multi-turn flow works)
    if not session_id:
        session_id = "timetable_default_session"
    
    msg = user_message.lower().strip()
    store = _load()
    state = _get_session(session_id) if session_id else {}
    step = state.get('step')
    print(
        "[TIMETABLE-TRACE] answer_timetable_query:enter | "
        f"session_id={session_id or ''} | "
        f"msg={msg[:120]} | "
        f"state_keys={sorted(list(state.keys()))} | "
        f"step={step or ''} | "
        f"remembered_class={state.get('remembered_class') or ''} | "
        f"store_count={len(store)}",
        flush=True,
    )

    # ── If class already remembered, answer directly ──────────────────────────
    remembered = state.get('remembered_class')
    if remembered and not step:
        intent_match = is_timetable_intent(msg)
        escape_match = is_timetable_escape(msg)
        followup_words = {'name', 'list', 'when', 'where', 'what', 'which',
                          'count', 'total', 'all', 'today', 'tomorrow', 'monday',
                          'tuesday', 'wednesday', 'thursday', 'friday', 'labs', 'free'}
        followup_match = msg.lower().strip() in followup_words
        print(
            "[TIMETABLE-TRACE] remembered_branch | "
            f"remembered={remembered} | "
            f"intent_match={intent_match} | "
            f"escape_match={escape_match} | "
            f"followup_match={followup_match}",
            flush=True,
        )
        # Farewell
        if re.match(r'^(bye|goodbye|quit|exit|see you|cya|later|thanks|thank you)[\s!.?]*$', msg):
            print("[TIMETABLE-TRACE] remembered_branch:return | reason=farewell", flush=True)
            _clear_session(session_id)
            return "<div>👋 Goodbye! Feel free to ask anytime. Have a great day!</div>"

        # Clear session on greetings
        if re.match(r'^(hi|hello|hey|good\s*(morning|evening|afternoon)|howdy)[\s!.?]*$', msg):
            print("[TIMETABLE-TRACE] remembered_branch:return | reason=greeting_clear_session", flush=True)
            _clear_session(session_id)
            return None

        # User correcting their class — restart guided flow
        if re.search(r'\b(no|wrong|actually|i am|i\'m)\b', msg, re.I) and \
           re.search(r'\b(1st|2nd|3rd|4th|5th|first|second|third|fourth|bca|btech|b\.tech|mca|cse|ece)\b', msg, re.I):
            print("[TIMETABLE-TRACE] remembered_branch:return | reason=class_correction", flush=True)
            _clear_session(session_id)
            state = {'step': 'ask_year', 'original_query': user_message}
            _set_session(session_id, state)
            return (
                "<div>No problem! Let me look up your correct timetable. 📅<br><br>"
                "Which <strong>year</strong> are you in?<br>"
                "<em>e.g. 1st, 2nd, 3rd, 4th</em></div>"
            )

        # "change timetable / switch class" — restart
        if re.search(r'\b(change|switch|different|another|other)\s+(class|timetable|section|year|branch)\b', msg, re.I):
            print("[TIMETABLE-TRACE] remembered_branch:return | reason=change_switch", flush=True)
            _clear_session(session_id)
            state = {'step': 'ask_year', 'original_query': user_message}
            _set_session(session_id, state)
            return (
                "<div>Sure! Let's find your timetable. 📅<br><br>"
                "Which <strong>year</strong> are you in?<br>"
                "<em>e.g. 1st, 2nd, 3rd, 4th</em></div>"
            )

        # Route to timetable only for actual timetable questions or short follow-up words
        # NOT for faculty info, mentor queries, general "tell me about X" questions
        FOLLOWUP_WORDS = {'name', 'list', 'when', 'where', 'what', 'which',
                          'count', 'total', 'all', 'today', 'tomorrow', 'monday',
                          'tuesday', 'wednesday', 'thursday', 'friday', 'labs', 'free'}
        print(
            "[TIMETABLE-TRACE] remembered_branch:route_check | "
            f"is_escape={escape_match} | "
            f"is_intent={intent_match} | "
            f"is_followup={followup_match}",
            flush=True,
        )
        if not is_timetable_escape(msg) and (
            is_timetable_intent(msg) or msg.lower().strip() in FOLLOWUP_WORDS
        ):
            print("[TIMETABLE-TRACE] remembered_branch:return | reason=answer_with_llm", flush=True)
            return _answer_with_llm(remembered, user_message, session_id, history=history)
        # Otherwise fall through to RAG (faculty, mentor, fees, etc.)
        print("[TIMETABLE-TRACE] remembered_branch:return | reason=fallthrough_none", flush=True)
        return None

    # ── Active session: continue guided flow ──────────────────────────────────
    if step == 'ask_year':
        print("[TIMETABLE-TRACE] active_session_branch | step=ask_year", flush=True)
        for y in ['1st', '2nd', '3rd', '4th', '5th', 'first', 'second', 'third', 'fourth']:
            # Word boundary match to avoid matching 'third' in phrases like 'third eye'
            pattern = r'\b' + re.escape(y) + r'\b'
            if re.search(pattern, msg, re.IGNORECASE):
                norm = {'first': '1st', 'second': '2nd', 'third': '3rd', 'fourth': '4th'}.get(y, y)
                state['year'] = norm
                break
        if 'year' not in state:
            print("[TIMETABLE-TRACE] active_session_branch:return | reason=ask_year_prompt", flush=True)
            return "<div>Please reply with your year — e.g. <strong>1st</strong>, <strong>2nd</strong>, <strong>3rd</strong>.</div>"
        state['step'] = 'ask_branch'
        _set_session(session_id, state)
        matches = _filter_classes(year=state['year'])
        print(
            "[TIMETABLE-TRACE] active_session_branch:return | reason=ask_branch_prompt | "
            f"matches={len(matches)}",
            flush=True,
        )
        return _ask_branch(_unique_branches(matches))

    if step == 'ask_branch':
        print("[TIMETABLE-TRACE] active_session_branch | step=ask_branch", flush=True)
        matched = _match_branch_from_reply(msg, state.get('year'))
        print(f"[TIMETABLE-TRACE] active_session_branch:match_branch | matched={matched or ''}", flush=True)
        if not matched:
            matches = _filter_classes(year=state.get('year'))
            print(
                "[TIMETABLE-TRACE] active_session_branch:return | reason=branch_prompt_again | "
                f"matches={len(matches)}",
                flush=True,
            )
            return f"<div>Please choose from:<br><strong>{', '.join(_unique_branches(matches))}</strong></div>"
        state['branch'] = matched
        state['step'] = 'ask_section'
        _set_session(session_id, state)
        matches = _filter_classes(year=state.get('year'), branch=matched)
        if len(matches) == 1:
            print("[TIMETABLE-TRACE] active_session_branch:return | reason=single_match_answer", flush=True)
            return _answer_with_llm(matches[0], state.get('original_query', user_message), session_id, history=history)
        print(
            "[TIMETABLE-TRACE] active_session_branch:return | reason=ask_section_prompt | "
            f"matches={len(matches)}",
            flush=True,
        )
        return _ask_section(_unique_sections(matches))

    if step == 'ask_section':
        print("[TIMETABLE-TRACE] active_session_branch | step=ask_section", flush=True)
        matches = _filter_classes(year=state.get('year'), branch=state.get('branch'))
        chosen = _match_section_from_reply(msg, matches)
        print(f"[TIMETABLE-TRACE] active_session_branch:match_section | chosen={chosen or ''}", flush=True)
        if not chosen:
            print(
                "[TIMETABLE-TRACE] active_session_branch:return | reason=section_prompt_again | "
                f"matches={len(matches)}",
                flush=True,
            )
            return f"<div>Please choose your section: <strong>{', '.join(_unique_sections(matches))}</strong></div>"
        print("[TIMETABLE-TRACE] active_session_branch:return | reason=answer_with_llm", flush=True)
        return _answer_with_llm(chosen, state.get('original_query', user_message), session_id, history=history)

    # ── No active session — check intent ──────────────────────────────────────
    # Handle farewell even without active session
    if re.match(r'^(bye|goodbye|quit|exit|see you|cya|later)[\s!.?]*$', msg):
        print("[TIMETABLE-TRACE] no_session_branch:return | reason=farewell", flush=True)
        return "<div>👋 Goodbye! Come back anytime. Have a great day!</div>"

    if not is_timetable_intent(msg):
        print("[TIMETABLE-TRACE] no_session_branch:return | reason=no_timetable_intent", flush=True)
        return None

    if not store:
        print("[TIMETABLE-TRACE] no_session_branch:return | reason=no_store", flush=True)
        return (
            "<div>No timetable uploaded yet. "
            "Ask an admin to upload via <strong>Admin Panel → Timetable</strong>.</div>"
        )

    # Try inline filters first
    inline = _extract_inline_filters(user_message)
    print(f"[TIMETABLE-TRACE] no_session_branch:inline_filters | inline={inline}", flush=True)
    if inline:
        matches = _filter_classes(**inline)
        print(f"[TIMETABLE-TRACE] no_session_branch:inline_matches | matches={len(matches)}", flush=True)
        if len(matches) == 1:
            print("[TIMETABLE-TRACE] no_session_branch:return | reason=inline_single_match", flush=True)
            return _answer_with_llm(matches[0], user_message, session_id)
        if len(matches) > 1:
            if 'year' in inline and 'branch' not in inline:
                state = {'step': 'ask_branch', 'year': inline['year'], 'original_query': user_message}
                _set_session(session_id, state)
                print("[TIMETABLE-TRACE] no_session_branch:return | reason=inline_ask_branch", flush=True)
                return _ask_branch(_unique_branches(matches))
            if 'year' in inline and 'branch' in inline:
                state = {'step': 'ask_section', 'year': inline['year'], 'branch': inline['branch'], 'original_query': user_message}
                _set_session(session_id, state)
                print("[TIMETABLE-TRACE] no_session_branch:return | reason=inline_ask_section", flush=True)
                return _ask_section(_unique_sections(matches))

    # Start guided flow
    state = {'step': 'ask_year', 'original_query': user_message}
    _set_session(session_id, state)
    print("[TIMETABLE-TRACE] no_session_branch:return | reason=start_guided_flow", flush=True)
    return (
        "<div>I can help with your timetable! 📅<br><br>"
        "Which <strong>year</strong> are you in?<br>"
        "<em>e.g. 1st, 2nd, 3rd, 4th</em></div>"
    )


# ── Centralized Query Router (Dispatcher Pattern) ────────────────────────────────

def route_query(msg: str, class_name: str, schedule: list, session_id: str = None, 
                history: list = None) -> tuple:
    """
    Centralized query router that determines query type and returns (handler_type, result).
    
    Handler types: 'display', 'count', 'search', 'time_lookup', 'llm_fallback'
    
    Returns: (handler_type, response_html)
    
    All routing logic flows through this single dispatcher.
    Preserves backward compatibility by reusing all existing handlers.
    
    Analytics: Records query execution (optional, non-blocking, never affects routing).
    """
    
    # ── Start performance timer ────────────────────────────────────────────────
    start_time = time.time()
    
    msg_lower = msg.lower().strip() if msg else ""
    
    try:
        # ── Display queries (highest priority due to deterministic rendering) ─────
        if _is_display_query(msg_lower):
            display_type = _detect_display_query_type(msg_lower)
            
            if display_type == 'full':
                if session_id:
                    _remember_query_context(session_id, 'display_full', class_name, msg)
                result = _render_full_table(class_name, schedule)
                # Log successful display query
                try:
                    _log_query(msg, 'display', _get_execution_time_ms(start_time), success=True)
                except Exception:
                    pass  # Analytics error never affects routing
                return ('display', result)
            
            elif display_type == 'today':
                day = _detect_day_from_msg(msg_lower)
                if not day:
                    from datetime import date
                    today = date.today()
                    day_index = today.weekday()
                    days = ["Mo", "Tu", "We", "Th", "Fr"]
                    day = days[day_index] if day_index < 5 else None
                if day:
                    if session_id:
                        _remember_query_context(session_id, 'display_today', class_name, msg, day=day)
                    result = _render_day_table(class_name, day, schedule)
                    # Log successful display query
                    try:
                        _log_query(msg, 'display', _get_execution_time_ms(start_time), success=True)
                    except Exception:
                        pass
                    return ('display', result)
            
            elif display_type == 'tomorrow':
                from datetime import date, timedelta
                tomorrow = date.today() + timedelta(days=1)
                day_index = tomorrow.weekday()
                days = ["Mo", "Tu", "We", "Th", "Fr"]
                day = days[day_index] if day_index < 5 else None
                if day:
                    if session_id:
                        _remember_query_context(session_id, 'display_tomorrow', class_name, msg, day=day)
                    result = _render_day_table(class_name, day, schedule)
                    # Log successful display query
                    try:
                        _log_query(msg, 'display', _get_execution_time_ms(start_time), success=True)
                    except Exception:
                        pass
                    return ('display', result)
            
            elif display_type in ["Mo", "Tu", "We", "Th", "Fr"]:
                if session_id:
                    _remember_query_context(session_id, 'display_day', class_name, msg, day=display_type)
                result = _render_day_table(class_name, display_type, schedule)
                # Log successful display query
                try:
                    _log_query(msg, 'display', _get_execution_time_ms(start_time), success=True)
                except Exception:
                    pass
                return ('display', result)
        
        # ── Count queries (deterministic calculation) ─────────────────────────────
        if _is_count_query(msg_lower):
            answer = _answer_count_query(class_name, msg, schedule)
            if answer:
                if session_id:
                    _, day = _detect_count_query_type(msg)
                    _remember_query_context(session_id, 'count', class_name, msg, day=day)
                # Log successful count query
                try:
                    _log_query(msg, 'count', _get_execution_time_ms(start_time), success=True)
                except Exception:
                    pass
                return ('count', answer)
        
        # ── Search queries (direct JSON lookup) ──────────────────────────────────
        if _is_search_query(msg_lower):
            answer = _answer_search_query(class_name, msg, schedule)
            if answer:
                if session_id:
                    _, target = _detect_search_query_type(msg)
                    _remember_query_context(session_id, 'search', class_name, msg)
                # Log successful search query
                try:
                    _log_query(msg, 'search', _get_execution_time_ms(start_time), success=True)
                except Exception:
                    pass
                return ('search', answer)
        
        # ── Time lookup queries (deterministic time calculation) ─────────────────
        if _is_time_lookup_query(msg_lower):
            answer = _answer_time_lookup_query(class_name, msg, schedule)
            if answer:
                if session_id:
                    lookup_type, target = _detect_time_lookup_type(msg)
                    _remember_query_context(session_id, 'time_lookup', class_name, msg)
                # Log successful time lookup query
                try:
                    _log_query(msg, 'time_lookup', _get_execution_time_ms(start_time), success=True)
                except Exception:
                    pass
                return ('time_lookup', answer)
        
        # ── Legacy display patterns (backward compatibility) ───────────────────────
        # Full timetable
        if any(p in msg_lower for p in ['full timetable', 'full schedule', 'show timetable', 'show schedule',
                                         'give me timetable', 'give timetable', 'entire timetable']):
            if session_id:
                _remember_query_context(session_id, 'display_full', class_name, msg)
            result = _render_full_table(class_name, schedule)
            # Log successful display query
            try:
                _log_query(msg, 'display', _get_execution_time_ms(start_time), success=True)
            except Exception:
                pass
            return ('display', result)
        
        # Day-specific timetable display
        day = _detect_day_from_msg(msg_lower)
        if day and any(p in msg_lower for p in ['timetable for', 'schedule for', 'timetable on', 'schedule on',
                                                'classes on', 'class on', 'monday timetable', 'tuesday timetable',
                                                'wednesday timetable', 'thursday timetable', 'friday timetable']):
            if session_id:
                _remember_query_context(session_id, 'display_day', class_name, msg, day=day)
            result = _render_day_table(class_name, day, schedule)
            # Log successful display query
            try:
                _log_query(msg, 'display', _get_execution_time_ms(start_time), success=True)
            except Exception:
                pass
            return ('display', result)
        
        # ── LLM fallback for complex/reasoning queries ──────────────────────────
        if session_id:
            _remember_query_context(session_id, 'llm_fallback', class_name, msg)
        
        tt = {'schedule': schedule}
        timetable_text = _timetable_to_text(tt)
        answer = _ask_llm(timetable_text, msg, history=history)
        
        if answer:
            # Log successful LLM fallback query
            try:
                _log_query(msg, 'llm_fallback', _get_execution_time_ms(start_time), success=True)
            except Exception:
                pass
            return ('llm_fallback', answer)
        
        result = _render_full_table(class_name, schedule)
        # Log successful LLM fallback query
        try:
            _log_query(msg, 'llm_fallback', _get_execution_time_ms(start_time), success=True)
        except Exception:
            pass
        return ('llm_fallback', result)
    
    except Exception as e:
        # Log exception with error details
        try:
            _log_query(msg, 'exception', _get_execution_time_ms(start_time), 
                      success=False, error_msg=str(e)[:100])
        except Exception:
            pass  # Analytics error never affects routing
        # Re-raise exception immediately (do NOT swallow)
        raise


# ── LLM answer ─────────────────────────────────────────────────────────────────

def _is_display_query(msg: str) -> bool:
    """Detect if query is a display-only request (should bypass LLM)."""
    if not msg:
        return False
    msg_lower = msg.lower()
    
    # Full timetable/schedule
    display_keywords = [
        'full timetable', 'full schedule', 'complete timetable', 'complete schedule',
        'entire timetable', 'entire schedule', 'weekly timetable', 'weekly schedule',
        'show me my timetable', 'show me my schedule', 'show my timetable', 'show my schedule',
        'show timetable', 'show schedule', 'display timetable', 'display schedule',
        'give me timetable', 'give timetable', 'give me schedule', 'give schedule',
        "today's timetable", "today's schedule", 'today timetable', 'today schedule',
        'tomorrow timetable', 'tomorrow schedule', "tomorrow's timetable", "tomorrow's schedule",
    ]
    
    # Specific weekday timetables
    weekday_keywords = [
        'monday timetable', 'monday schedule',
        'tuesday timetable', 'tuesday schedule',
        'wednesday timetable', 'wednesday schedule',
        'thursday timetable', 'thursday schedule',
        'friday timetable', 'friday schedule',
        'timetable for monday', 'schedule for monday',
        'timetable for tuesday', 'schedule for tuesday',
        'timetable for wednesday', 'schedule for wednesday',
        'timetable for thursday', 'schedule for thursday',
        'timetable for friday', 'schedule for friday',
    ]
    
    # Check if message contains any display keyword
    for keyword in display_keywords + weekday_keywords:
        if keyword in msg_lower:
            return True
    
    return False


def _detect_display_query_type(msg: str) -> str:
    """Detect which type of display query: 'full', 'today', 'tomorrow', or specific day."""
    if not msg:
        return None
    msg_lower = msg.lower()
    
    # Full/weekly
    if any(k in msg_lower for k in ['full', 'complete', 'entire', 'weekly', 'show timetable', 'show schedule', 'display', 'give']):
        return 'full'
    
    # Today
    if any(k in msg_lower for k in ["today's", 'today timetable', 'today schedule']):
        return 'today'
    
    # Tomorrow
    if any(k in msg_lower for k in ["tomorrow's", 'tomorrow timetable', 'tomorrow schedule']):
        return 'tomorrow'
    
    # Specific weekday
    for day_abbr, day_full in DAY_FULL.items():
        if day_full.lower() in msg_lower or day_abbr.lower() in msg_lower:
            return day_abbr
    
    return None


def _is_count_query(msg: str) -> bool:
    """Detect if query is a count-only request (should bypass LLM)."""
    if not msg:
        return False
    msg_lower = msg.lower()
    
    count_keywords = [
        'how many classes', 'how many periods', 'how many labs',
        'total classes', 'total periods', 'total labs',
        'how many free', 'how many gaps', 'how many breaks',
        'total free', 'total gaps', 'total breaks',
        'count classes', 'count periods', 'count free',
    ]
    
    for keyword in count_keywords:
        if keyword in msg_lower:
            return True
    
    return False


def _detect_count_query_type(msg: str) -> tuple:
    """Detect count query type and target. Returns (query_type, day_abbr or 'week' or None)."""
    if not msg:
        return None, None
    msg_lower = msg.lower()
    
    # Determine if counting classes or free periods
    is_free = any(k in msg_lower for k in ['free', 'gap', 'break', 'vacant', 'empty'])
    query_type = 'free' if is_free else 'classes'
    
    # Determine target using unified normalization
    if 'today' in msg_lower:
        today_idx = datetime.now().weekday()
        if today_idx <= 4:
            return query_type, WEEKDAY_TO_ABBR[today_idx]
        return query_type, None
    
    if 'tomorrow' in msg_lower:
        tomorrow_idx = datetime.now().weekday() + 1
        if tomorrow_idx <= 4:
            return query_type, WEEKDAY_TO_ABBR[tomorrow_idx]
        return query_type, None
    
    if any(k in msg_lower for k in ['this week', 'week']):
        return query_type, 'week'
    
    # Try to detect specific day
    day = _normalize_day(msg)
    if day:
        return query_type, day
    
    # Default to total for entire schedule
    return query_type, 'all'


def _format_count_answer(query_type: str, target: str, count: int, class_name: str) -> str:
    """Format count query answer as HTML."""
    if query_type == 'free':
        item_name = 'free period' if count == 1 else 'free periods'
    else:
        item_name = 'class' if count == 1 else 'classes'
    
    if target == 'today':
        period = "Today"
    elif target == 'tomorrow':
        period = "Tomorrow"
    elif target == 'week':
        period = "This week"
    elif target == 'all':
        period = "Total"
    else:
        period = f"{DAY_FULL.get(target, target)}"
    
    return (
        f"<div style='color:#c4b5fd;font-size:14px'>"
        f"<strong>{class_name}</strong> — {period}: "
        f"<span style='color:#a78bfa;font-weight:600;font-size:16px'>{count}</span> {item_name}"
        f"</div>"
    )


def _answer_count_query(class_name: str, query: str, schedule: list) -> str:
    """Answer count queries directly without LLM."""
    query_type, target = _detect_count_query_type(query)
    
    if target == 'today':
        from datetime import date
        today = date.today()
        day_index = today.weekday()
        days = ["Mo", "Tu", "We", "Th", "Fr"]
        day = days[day_index] if day_index < 5 else None
        if day:
            count = count_free_slots(schedule, day) if query_type == 'free' else count_classes(schedule, day)
            return _format_count_answer(query_type, target, count, class_name)
    
    elif target == 'tomorrow':
        from datetime import date, timedelta
        tomorrow = date.today() + timedelta(days=1)
        day_index = tomorrow.weekday()
        days = ["Mo", "Tu", "We", "Th", "Fr"]
        day = days[day_index] if day_index < 5 else None
        if day:
            count = count_free_slots(schedule, day) if query_type == 'free' else count_classes(schedule, day)
            return _format_count_answer(query_type, target, count, class_name)
    
    elif target == 'week':
        count = count_free_slots(schedule) if query_type == 'free' else count_classes(schedule)
        return _format_count_answer(query_type, target, count, class_name)
    
    elif target == 'all':
        count = count_free_slots(schedule) if query_type == 'free' else count_classes(schedule)
        return _format_count_answer(query_type, target, count, class_name)
    
    elif target in ["Mo", "Tu", "We", "Th", "Fr"]:
        count = count_free_slots(schedule, target) if query_type == 'free' else count_classes(schedule, target)
        return _format_count_answer(query_type, target, count, class_name)
    
    # Fallback if detection fails
    return None


def _is_search_query(msg: str) -> bool:
    """Detect if query is a search-only request (should bypass LLM)."""
    if not msg:
        return False
    msg_lower = msg.lower()
    
    search_keywords = [
        'who teach', 'which teacher', 'faculty for', 'teacher for',
        'what subject', 'which subject', 'all subject', 'list subject', 'show subject',
        'which room', 'what room', 'classroom for', 'room for',
        'list all room', 'show all room', 'all room',
        'list all faculty', 'list all teacher', 'show all teacher', 'show all faculty',
    ]
    
    for keyword in search_keywords:
        if keyword in msg_lower:
            return True
    
    return False


def _detect_search_query_type(msg: str) -> tuple:
    """Detect search query type. Returns (search_type, target_name or None)."""
    if not msg:
        return None, None
    msg_lower = msg.lower()
    
    # Faculty queries
    if any(k in msg_lower for k in ['who teach', 'which teacher', 'faculty for', 'teacher for']):
        # Extract subject name
        for keyword in ['who teach', 'which teacher', 'faculty for', 'teacher for']:
            if keyword in msg_lower:
                idx = msg_lower.find(keyword)
                target = msg[idx + len(keyword):].strip('? .,').strip()
                return 'faculty', target if target else None
    
    # Subject listing queries
    if any(k in msg_lower for k in ["what subject", 'which subject', 'all subject', 'list subject', 'show subject']):
        return 'subjects', None
    
    # Room queries
    if any(k in msg_lower for k in ['which room', 'what room', 'classroom for', 'room for']):
        # Extract subject name
        for keyword in ['which room', 'what room', 'classroom for', 'room for']:
            if keyword in msg_lower:
                idx = msg_lower.find(keyword)
                target = msg[idx + len(keyword):].strip('? .,').strip()
                return 'room', target if target else None
    
    # Room listing
    if any(k in msg_lower for k in ['list all room', 'show all room', 'all room']):
        return 'rooms', None
    
    # Faculty listing
    if any(k in msg_lower for k in ['list all faculty', 'list all teacher', 'show all teacher', 'show all faculty']):
        return 'faculty_list', None
    
    return None, None


def _format_search_answer(search_type: str, data: str, class_name: str = None) -> str:
    """Format search query answer as HTML."""
    if search_type == 'faculty':
        return (
            f"<div style='color:#c4b5fd;font-size:14px'>"
            f"<strong>{class_name or 'Timetable'}</strong>:<br>"
            f"<span style='color:#a78bfa'>{data}</span>"
            f"</div>"
        )
    elif search_type == 'room':
        return (
            f"<div style='color:#c4b5fd;font-size:14px'>"
            f"<strong>{class_name or 'Timetable'}</strong>:<br>"
            f"<span style='color:#a78bfa'>{data}</span>"
            f"</div>"
        )
    elif search_type == 'subjects':
        items = data.split(', ') if data else []
        if not items or not items[0]:
            return f"<div style='color:#cbd5e1'>No subjects found.</div>"
        html_items = ''.join(f"<li style='color:#a78bfa'>{item}</li>" for item in items)
        return f"<div style='color:#c4b5fd'><ul style='margin:8px 0'>{html_items}</ul></div>"
    elif search_type == 'rooms':
        items = data.split(', ') if data else []
        if not items or not items[0]:
            return f"<div style='color:#cbd5e1'>No rooms found.</div>"
        html_items = ''.join(f"<li style='color:#a78bfa'>{item}</li>" for item in items)
        return f"<div style='color:#c4b5fd'><ul style='margin:8px 0'>{html_items}</ul></div>"
    elif search_type == 'faculty_list':
        items = data.split(', ') if data else []
        if not items or not items[0]:
            return f"<div style='color:#cbd5e1'>No faculty found.</div>"
        html_items = ''.join(f"<li style='color:#a78bfa'>{item}</li>" for item in items)
        return f"<div style='color:#c4b5fd'><ul style='margin:8px 0'>{html_items}</ul></div>"
    
    return f"<div style='color:#c4b5fd'>{data}</div>"


def _answer_search_query(class_name: str, query: str, schedule: list) -> str:
    """Answer search queries directly without LLM."""
    # Try fuzzy correction on subject if present
    if any(word in query.lower() for word in ['teach', 'faculty', 'room', 'which']):
        corrected_query, correction_action, confidence = _correct_subject_in_query(query, schedule)
        if correction_action == 'auto_correct':
            query = corrected_query  # Use corrected query
    
    search_type, target = _detect_search_query_type(query)
    
    if search_type == 'faculty':
        # Who teaches <subject>?
        if not target:
            # No subject specified - ask for clarification
            return _ask_subject_clarification(schedule)
        
        # Try to resolve via alias first
        resolved_subject = _resolve_subject_via_alias(target, schedule)
        if resolved_subject:
            target = resolved_subject
        else:
            # Try fuzzy match on target subject
            corrected_subject, confidence, matches = _fuzzy_match_subject(schedule, target)
            if corrected_subject and confidence >= 0.65:
                action, value = _apply_fuzzy_correction(corrected_subject, confidence, matches)
                if action == 'auto_correct':
                    target = corrected_subject
                elif action == 'clarify':
                    # Multiple similar subjects - use clarification
                    question = f"Did you mean one of these? Or which subject?"
                    return _format_clarification(question, value)
        
        matches = find_subject(schedule, target)
        if matches:
            faculty_list = set()
            for match in matches:
                faculty = (match.get('faculty') or '').strip()
                if faculty and faculty.upper() != 'FREE':
                    faculty_list.add(faculty)
            if faculty_list:
                result = ', '.join(sorted(faculty_list))
                return _format_search_answer('faculty', f"Teaching {target}: {result}", class_name)
        
        # No exact match - ask for clarification with available subjects
        return _ask_subject_clarification(schedule)
    
    elif search_type == 'subjects':
        # What subjects do I have?
        subjects = get_all_subjects(schedule)
        if subjects:
            result = ', '.join(subjects)
            return _format_search_answer('subjects', result, class_name)
        return f"<div style='color:#cbd5e1'>No subjects found.</div>"
    
    elif search_type == 'room':
        # Which room is <subject>?
        if not target:
            # No subject specified - ask for clarification
            return _ask_room_clarification(schedule)
        
        # Try to resolve via alias first
        resolved_subject = _resolve_subject_via_alias(target, schedule)
        if resolved_subject:
            target = resolved_subject
        else:
            # Try fuzzy match on target subject
            corrected_subject, confidence, matches = _fuzzy_match_subject(schedule, target)
            if corrected_subject and confidence >= 0.65:
                action, value = _apply_fuzzy_correction(corrected_subject, confidence, matches)
                if action == 'auto_correct':
                    target = corrected_subject
                elif action == 'clarify':
                    question = f"Did you mean one of these? Or which subject?"
                    return _format_clarification(question, value)
        
        matches = find_subject(schedule, target)
        if matches:
            rooms = set()
            for match in matches:
                room = (match.get('room') or '').strip()
                if room and room.upper() != 'FREE':
                    rooms.add(room)
            if rooms:
                result = ', '.join(sorted(rooms))
                return _format_search_answer('room', f"Rooms for {target}: {result}", class_name)
        
        # No exact match - ask for clarification with available subjects
        return _ask_room_clarification(schedule)
    
    elif search_type == 'rooms':
        # Show all rooms
        rooms = get_all_rooms(schedule)
        if rooms:
            result = ', '.join(rooms)
            return _format_search_answer('rooms', result, class_name)
        return f"<div style='color:#cbd5e1'>No rooms found.</div>"
    
    elif search_type == 'faculty_list':
        # List all faculty
        faculty = get_all_faculty(schedule)
        if faculty:
            result = ', '.join(faculty)
            return _format_search_answer('faculty_list', result, class_name)
        return f"<div style='color:#cbd5e1'>No faculty found.</div>"
    
    # Fallback
    return None


def _is_time_lookup_query(msg: str) -> bool:
    """Detect if query is a time lookup request (should bypass LLM)."""
    if not msg:
        return False
    msg_lower = msg.lower()
    
    time_keywords = [
        'first class', 'last class', 'next class',
    ]
    
    for keyword in time_keywords:
        if keyword in msg_lower:
            return True
    
    return False


def _detect_time_lookup_type(msg: str) -> tuple:
    """Detect time lookup query type. Returns (lookup_type, day_str or None)."""
    if not msg:
        return None, None
    msg_lower = msg.lower()
    
    # First class today/on <day>
    if 'first class' in msg_lower:
        day = _detect_day_from_msg(msg)
        if day:
            return 'first_on_day', day
        else:
            # Default to today
            from datetime import date
            today = date.today()
            day_index = today.weekday()
            days = ["Mo", "Tu", "We", "Th", "Fr"]
            return 'first_on_day', days[day_index] if day_index < 5 else None
    
    # Last class today/on <day>
    if 'last class' in msg_lower:
        day = _detect_day_from_msg(msg)
        if day:
            return 'last_on_day', day
        else:
            # Default to today
            from datetime import date
            today = date.today()
            day_index = today.weekday()
            days = ["Mo", "Tu", "We", "Th", "Fr"]
            return 'last_on_day', days[day_index] if day_index < 5 else None
    
    # Next class
    if 'next class' in msg_lower:
        return 'next', None
    
    return None, None


def _parse_time_string(time_str: str) -> tuple:
    """Parse time string like '11 AM', '2 PM', '14:30' to (hour, minute, period)."""
    if not time_str:
        return None, None, None
    
    time_str = time_str.strip().upper()
    
    # Handle formats like "11 AM", "2 PM", "11AM", "2PM"
    import re
    match = re.match(r'(\d{1,2})\s*(?::(\d{2}))?\s*(AM|PM)?', time_str)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2)) if match.group(2) else 0
        period = match.group(3) or 'AM'
        
        # Convert to 24-hour format
        if period == 'PM' and hour != 12:
            hour += 12
        elif period == 'AM' and hour == 12:
            hour = 0
        
        return hour, minute, period
    
    # Handle 24-hour format like "14:30"
    match = re.match(r'(\d{1,2}):(\d{2})', time_str)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))
        return hour, minute, None
    
    return None, None, None


def _format_time_answer(lookup_type: str, data: str, class_name: str = None) -> str:
    """Format time lookup answer as HTML."""
    return (
        f"<div style='color:#c4b5fd;font-size:14px'>"
        f"<strong>{class_name or 'Timetable'}</strong>:<br>"
        f"<span style='color:#a78bfa'>{data}</span>"
        f"</div>"
    )


def _answer_time_lookup_query(class_name: str, query: str, schedule: list) -> str:
    """Answer time lookup queries directly without LLM."""
    lookup_type, target = _detect_time_lookup_type(query)
    
    if lookup_type == 'first_on_day':
        # First class on <day>
        if target:
            first_class = get_first_class(schedule, target)
            if first_class and first_class.get('course', '').upper() != 'FREE':
                subject = first_class.get('course', '').strip()
                time_str = first_class.get('time', '').strip()
                room = first_class.get('room', '').strip()
                faculty = first_class.get('faculty', '').strip()
                result = f"First class on {target}: {subject} at {time_str} ({faculty}, {room})"
                return _format_time_answer('first_on_day', result, class_name)
            # Convert abbreviated day to full name for display
            day_full = DAY_FULL.get(target, target)
            return f"<div style='color:#cbd5e1'>No classes on <strong>{day_full}</strong> for <strong>{class_name}</strong>.</div>"
    
    elif lookup_type == 'last_on_day':
        # Last class on <day>
        if target:
            last_class = get_last_class(schedule, target)
            if last_class and last_class.get('course', '').upper() != 'FREE':
                subject = last_class.get('course', '').strip()
                time_str = last_class.get('time', '').strip()
                room = last_class.get('room', '').strip()
                faculty = last_class.get('faculty', '').strip()
                result = f"Last class on {target}: {subject} at {time_str} ({faculty}, {room})"
                return _format_time_answer('last_on_day', result, class_name)
            # Convert abbreviated day to full name for display
            day_full = DAY_FULL.get(target, target)
            return f"<div style='color:#cbd5e1'>No classes on <strong>{day_full}</strong> for <strong>{class_name}</strong>.</div>"
    
    elif lookup_type == 'next':
        # Next class (today onwards)
        from datetime import date
        today = date.today()
        day_index = today.weekday()
        days = ["Mo", "Tu", "We", "Th", "Fr"]
        
        # Search from today onwards
        for i in range(day_index, 5):
            day = days[i]
            first_class = get_first_class(schedule, day)
            if first_class and first_class.get('course', '').upper() != 'FREE':
                subject = first_class.get('course', '').strip()
                time_str = first_class.get('time', '').strip()
                room = first_class.get('room', '').strip()
                faculty = first_class.get('faculty', '').strip()
                result = f"Next class: {subject} on {day} at {time_str} ({faculty}, {room})"
                return _format_time_answer('next', result, class_name)
        
        return f"<div style='color:#cbd5e1'>No upcoming classes this week</div>"
    
    # Fallback (for 'time' lookup type - not supported with current data structure)
    return None


# ── Follow-up Conversation Memory (lightweight, Module 2 only) ─────────────────

def _get_session_context(session_id: str) -> dict:
    """Get conversation memory for this session."""
    if not session_id:
        return {}
    
    sessions = _load_sessions()
    session = sessions.get(session_id, {})
    
    # Ensure memory key exists
    if 'timetable_memory' not in session:
        session['timetable_memory'] = {}
    
    return session.get('timetable_memory', {})


def _update_session_memory(session_id: str, updates: dict):
    """Update conversation memory (lightweight metadata only)."""
    if not session_id:
        return
    
    sessions = _load_sessions()
    if session_id not in sessions:
        sessions[session_id] = {}
    
    # Ensure memory key exists
    if 'timetable_memory' not in sessions[session_id]:
        sessions[session_id]['timetable_memory'] = {}
    
    # Merge updates
    sessions[session_id]['timetable_memory'].update(updates)
    _save_sessions(sessions)


def _extract_subject_from_query(query: str) -> str:
    """Extract subject name from query like 'Who teaches DBMS?'."""
    msg_lower = query.lower()
    
    # Remove question marks
    query_clean = query.strip('?').strip()
    
    # Try to extract after known keywords
    for keyword in ['teaches', 'who teach', 'faculty for', 'teacher for', 'room for', 'classroom for', 'which room']:
        idx = msg_lower.find(keyword)
        if idx != -1:
            start = idx + len(keyword)
            subject = query_clean[start:].strip()
            if subject:
                return subject
    
    return None


def _extract_faculty_from_query(query: str) -> str:
    """Extract faculty/teacher name from query."""
    msg_lower = query.lower()
    
    # Remove question marks and punctuation
    query_clean = query.strip('?').strip().rstrip('.,;:!\'\"')
    
    # Look for patterns like "Faculty for <name>" or "Teacher for <name>"
    for keyword in ['faculty', 'teacher', 'prof', 'faculty:']:
        if keyword in msg_lower:
            idx = msg_lower.rfind(keyword)
            if idx != -1:
                start = idx + len(keyword)
                extracted = query_clean[start:].strip()
                if extracted and not any(word in extracted.lower() for word in ['what', 'which', 'any', 'do i', 'today', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday']):
                    return extracted
    
    return None


def _extract_room_from_query(query: str) -> str:
    """Extract room name from query."""
    msg_lower = query.lower()
    
    # Remove question marks and punctuation
    query_clean = query.strip('?').strip().rstrip('.,;:!\'\"')
    
    # Look for patterns like "Room for <name>" or "Room <name>"
    for keyword in ['room', 'classroom']:
        if keyword in msg_lower:
            idx = msg_lower.rfind(keyword)
            if idx != -1:
                start = idx + len(keyword)
                extracted = query_clean[start:].strip()
                if extracted and not any(word in extracted.lower() for word in ['what', 'which', 'any', 'do i', 'today', 'all']):
                    return extracted
    
    return None


def _remember_query_context(session_id: str, query_type: str, class_name: str, query: str, day: str = None, classes_data: str = None):
    """Remember query context for follow-up queries."""
    if not session_id:
        return
    
    memory_update = {
        'last_query_type': query_type,
        'last_class_name': class_name,
        'last_full_query': query,
    }
    
    # Store day context if available
    if day:
        memory_update['last_day'] = day
    
    # Extract and store subject/faculty/room if found
    subject = _extract_subject_from_query(query)
    if subject:
        memory_update['last_subject'] = subject
    
    faculty = _extract_faculty_from_query(query)
    if faculty:
        memory_update['last_faculty'] = faculty
    
    room = _extract_room_from_query(query)
    if room:
        memory_update['last_room'] = room
    
    # Store lightweight classes data if provided (just count and subjects)
    if classes_data:
        memory_update['last_classes_info'] = classes_data
    
    _update_session_memory(session_id, memory_update)


def _resolve_follow_up_query(session_id: str, query: str, schedule: list, class_name: str) -> tuple:
    """
    Resolve follow-up queries using conversation memory.
    Returns (should_use_direct_answer, resolved_query, resolved_day, resolved_subject, resolved_faculty, resolved_room)
    If should_use_direct_answer is True, the resolved values provide context for answering.
    """
    if not session_id:
        return False, query, None, None, None, None
    
    memory = _get_session_context(session_id)
    msg_lower = query.lower()
    
    # Detect follow-up patterns
    
    # Follow-up 1: "Who teaches the first class?" after showing a day's timetable
    if any(word in msg_lower for word in ['who teach', 'faculty', 'teacher']) and memory.get('last_day'):
        day = memory.get('last_day')
        first_class = get_first_class(schedule, day)
        if first_class:
            subject = first_class.get('subject', '').strip()
            faculty = first_class.get('faculty', '').strip()
            if subject and faculty and faculty.upper() != 'FREE':
                memory_update = {'last_subject': subject, 'last_faculty': faculty}
                _update_session_memory(session_id, memory_update)
                return True, f"Who teaches {subject}?", day, subject, faculty, None
    
    # Follow-up 2: "Any free periods?" after showing a day's timetable
    if any(word in msg_lower for word in ['free', 'gap', 'break']) and memory.get('last_day'):
        day = memory.get('last_day')
        return True, f"How many free periods on {day}?", day, None, None, None
    
    # Follow-up 3: "Which room?" after asking about a subject
    if any(word in msg_lower for word in ['which room', 'what room', 'room for', 'classroom']) and memory.get('last_subject'):
        subject = memory.get('last_subject')
        return True, f"Which room is {subject}?", None, subject, None, None
    
    # Follow-up 4: "Faculty?" after asking about first/last class
    if any(word in msg_lower for word in ['faculty', 'teacher']) and memory.get('last_subject') and not memory.get('last_faculty'):
        subject = memory.get('last_subject')
        day = memory.get('last_day')
        return True, f"Who teaches {subject}?", day, subject, None, None
    
    # Follow-up 5: "Room?" after asking about first/last class
    if any(word in msg_lower for word in ['room', 'classroom']) and memory.get('last_subject') and not memory.get('last_room'):
        subject = memory.get('last_subject')
        return True, f"Which room is {subject}?", None, subject, None, None
    
    # No follow-up context found
    return False, query, None, None, None, None


# ── Smart Clarification Engine (Module 2 only) ──────────────────────────────

def _needs_subject_clarification(schedule: list, query: str) -> tuple:
    """Check if query needs subject clarification. Returns (needs_clarification, subjects_list)."""
    if not schedule:
        return False, []
    
    # Queries that may need subject clarification
    needs_clarif = any(word in query.lower() for word in ['faculty', 'teacher', 'room', 'which room', 'what room'])
    if not needs_clarif:
        return False, []
    
    # Get all subjects
    subjects = get_all_subjects(schedule)
    
    # If only one subject exists, no clarification needed
    if len(subjects) <= 1:
        return False, subjects
    
    # Multiple subjects - clarification needed
    return True, subjects


def _needs_section_clarification(store: dict, year: str = None, branch: str = None) -> tuple:
    """Check if multiple sections exist. Returns (needs_clarification, sections_list)."""
    if not store:
        return False, []
    
    classes = list(store.keys())
    
    # Filter by year/branch if provided
    if year or branch:
        filtered = _filter_classes(year=year, branch=branch)
        classes = filtered
    
    # If only one class matches, no clarification needed
    if len(classes) <= 1:
        return False, classes
    
    # Multiple classes - need section clarification
    sections = _unique_sections(classes)
    return len(sections) > 1, sections


def _format_clarification(question: str, options: list) -> str:
    """Format clarification message with options as HTML list."""
    if not options:
        return f"<div>{question}</div>"
    
    # Create HTML list
    items = ''.join(f"<li style='color:#a78bfa'>{opt}</li>" for opt in options)
    return (
        f"<div style='color:#c4b5fd'>"
        f"<strong>{question}</strong><br>"
        f"<ul style='margin:8px 0 0 16px;color:#a78bfa'>{items}</ul>"
        f"</div>"
    )


def _ask_subject_clarification(schedule: list, query_context: str = "") -> str:
    """Ask user which subject they're referring to."""
    subjects = get_all_subjects(schedule)
    if len(subjects) <= 1:
        return None  # No clarification needed
    
    question = "Which subject are you referring to?"
    return _format_clarification(question, subjects)


def _ask_section_clarification(sections: list) -> str:
    """Ask user which section they're in."""
    if len(sections) <= 1:
        return None  # No clarification needed
    
    question = "Which section are you in?"
    return _format_clarification(question, sections)


def _ask_day_clarification(query_context: str = "") -> str:
    """Ask user which day they're asking about."""
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    question = "Which day are you asking about?"
    return _format_clarification(question, days)


def _ask_room_clarification(schedule: list) -> str:
    """Ask user which subject's room they want to know."""
    subjects = get_all_subjects(schedule)
    if len(subjects) <= 1:
        return None
    
    question = "Which subject's room do you want to know?"
    return _format_clarification(question, subjects)


def _should_clarify_search_query(schedule: list, search_type: str, target: str) -> tuple:
    """
    Determine if a search query needs clarification.
    Returns (should_clarify, clarification_message, resolved_target)
    """
    if not target:
        # No target specified - need clarification
        if search_type == 'faculty':
            return True, _ask_subject_clarification(schedule), None
        elif search_type == 'room':
            return True, _ask_room_clarification(schedule), None
    
    # Target specified - check if it's unique
    if search_type == 'faculty':
        matches = find_subject(schedule, target)
        if not matches:
            # Subject not found - need clarification
            return True, _ask_subject_clarification(schedule), None
        if len(matches) > 1:
            # Multiple matches - could clarify, but search handles it
            pass
    
    elif search_type == 'room':
        matches = find_subject(schedule, target)
        if not matches:
            return True, _ask_room_clarification(schedule), None
    
    # No clarification needed
    return False, None, target


def _should_clarify_display_query(store: dict, year: str = None, branch: str = None) -> tuple:
    """
    Determine if display query needs section clarification.
    Returns (should_clarify, clarification_message, resolved_section)
    """
    needs_section, sections = _needs_section_clarification(store, year, branch)
    
    if needs_section and sections:
        clarif_msg = _ask_section_clarification(sections)
        return True, clarif_msg, None
    
    return False, None, None


# ── Fuzzy Matching Engine (Module 2 only) ──────────────────────────────────

def _fuzzy_match(query: str, candidates: list, threshold: float = 0.6) -> tuple:
    """
    Fuzzy match query against candidates using difflib.
    Returns (best_match, confidence, all_matches)
    
    confidence > 0.8 → high confidence (auto-correct)
    0.6 < confidence <= 0.8 → medium confidence (clarify)
    confidence <= 0.6 → low confidence (keep original)
    """
    if not query or not candidates:
        return None, 0.0, []
    
    query_lower = query.lower().strip()
    matches = []
    
    for candidate in candidates:
        candidate_lower = candidate.lower().strip()
        if candidate_lower == query_lower:
            # Exact match
            return candidate, 1.0, [(candidate, 1.0)]
        
        # Calculate similarity ratio
        ratio = SequenceMatcher(None, query_lower, candidate_lower).ratio()
        if ratio >= threshold:
            matches.append((candidate, ratio))
    
    if not matches:
        return None, 0.0, []
    
    # Sort by confidence (highest first)
    matches.sort(key=lambda x: x[1], reverse=True)
    best_match, best_confidence = matches[0]
    
    return best_match, best_confidence, matches


def _fuzzy_match_weekday(query: str) -> tuple:
    """
    Fuzzy match weekday with typo tolerance.
    Returns (corrected_day, confidence, all_matches)
    Handles: Monday, Tuesday, Wednesday, Thursday, Friday
    Abbreviations: Mo, Tu, We, Th, Fr
    """
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    abbr_map = {"Mo": "Monday", "Tu": "Tuesday", "We": "Wednesday", "Th": "Thursday", "Fr": "Friday"}
    
    if not query:
        return None, 0.0, []
    
    query_lower = query.lower().strip()
    
    # Check abbreviations first
    for abbr, full in abbr_map.items():
        if abbr.lower() == query_lower:
            return abbr, 1.0, [(abbr, 1.0)]
    
    # Fuzzy match against full names
    best_match, confidence, matches = _fuzzy_match(query, weekdays, threshold=0.6)
    
    return best_match, confidence, matches


def _fuzzy_match_subject(schedule: list, query: str) -> tuple:
    """
    Fuzzy match subject name with typo tolerance.
    Returns (corrected_subject, confidence, all_matches)
    """
    subjects = get_all_subjects(schedule)
    if not subjects:
        return None, 0.0, []
    
    return _fuzzy_match(query, subjects, threshold=0.6)


def _fuzzy_match_faculty(schedule: list, query: str) -> tuple:
    """
    Fuzzy match faculty name with typo tolerance.
    Returns (corrected_faculty, confidence, all_matches)
    """
    faculty_list = get_all_faculty(schedule)
    if not faculty_list:
        return None, 0.0, []
    
    return _fuzzy_match(query, faculty_list, threshold=0.6)


def _fuzzy_match_room(schedule: list, query: str) -> tuple:
    """
    Fuzzy match room code with typo tolerance.
    Returns (corrected_room, confidence, all_matches)
    """
    rooms = get_all_rooms(schedule)
    if not rooms:
        return None, 0.0, []
    
    return _fuzzy_match(query, rooms, threshold=0.6)


def _apply_fuzzy_correction(best_match: str, confidence: float, all_matches: list) -> tuple:
    """
    Determine how to handle fuzzy match result.
    Returns (action, value)
    
    Actions:
      'auto_correct' → Use best_match automatically
      'clarify' → Show multiple options
      'keep_original' → Keep user input as-is
    """
    if confidence >= 0.85:
        # High confidence - auto-correct
        return 'auto_correct', best_match
    
    elif confidence >= 0.65 and len(all_matches) > 1:
        # Medium confidence with multiple matches - clarify
        return 'clarify', [match[0] for match in all_matches]
    
    elif confidence >= 0.65 and len(all_matches) == 1:
        # Medium confidence with single match - auto-correct
        return 'auto_correct', best_match
    
    else:
        # Low confidence - keep original
        return 'keep_original', None


def _correct_subject_in_query(query: str, schedule: list) -> tuple:
    """
    Attempt to correct subject name in query.
    Returns (corrected_query, action, confidence)
    
    Looks for subject names in query and corrects typos.
    """
    if not query or not schedule:
        return query, 'none', 0.0
    
    # Try to find subject patterns in query
    subjects = get_all_subjects(schedule)
    if not subjects:
        return query, 'none', 0.0
    
    query_lower = query.lower()
    
    # Find longest subject match attempt
    best_subject = None
    best_confidence = 0.0
    best_action = 'none'
    
    for subject in subjects:
        subject_lower = subject.lower()
        if subject_lower in query_lower:
            # Exact substring match - no correction needed
            return query, 'none', 1.0
    
    # Try fuzzy match
    for subject in subjects:
        match, conf, _ = _fuzzy_match(subject, [w for w in query.split() if len(w) > 2], threshold=0.6)
        if conf > best_confidence:
            best_confidence = conf
            best_subject = subject
    
    if best_subject:
        action, value = _apply_fuzzy_correction(best_subject, best_confidence, [])
        if action == 'auto_correct':
            corrected = query.replace([w for w in query.split() if w.lower() == best_subject.lower().split()[0]][0], best_subject, 1)
            return corrected, action, best_confidence
    
    return query, 'none', 0.0


def _correct_weekday_in_query(query: str) -> tuple:
    """
    Attempt to correct weekday name in query.
    Returns (corrected_query, action, confidence)
    """
    if not query:
        return query, 'none', 0.0
    
    query_lower = query.lower()
    weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday"]
    
    # Check for exact match first
    for day in weekdays:
        if day in query_lower:
            return query, 'none', 1.0
    
    # Try to find typos in words
    words = query.split()
    for i, word in enumerate(words):
        corrected, confidence, matches = _fuzzy_match_weekday(word)
        if corrected and confidence >= 0.65:
            action, value = _apply_fuzzy_correction(corrected, confidence, matches)
            if action == 'auto_correct':
                corrected_query = ' '.join(words[:i] + [corrected] + words[i+1:])
                return corrected_query, action, confidence
    
    return query, 'none', 0.0






def _answer_with_llm(class_name: str, original_query: str, session_id: str, history: list = None) -> str:
    """Resolve class → answer query. Delegates to centralized router."""
    if session_id:
        _set_session(session_id, {'remembered_class': class_name})

    store = _load()
    tt = store.get(class_name)
    if not tt:
        return f"<div>Timetable not found for <strong>{class_name}</strong>.</div>"

    schedule = tt.get('schedule', [])
    msg = original_query.lower()
    
    # ── Check for follow-up queries using conversation memory ─────────────────
    if session_id:
        should_use_followup, resolved_query, resolved_day, resolved_subject, resolved_faculty, resolved_room = \
            _resolve_follow_up_query(session_id, original_query, schedule, class_name)
        if should_use_followup:
            msg = resolved_query.lower()
            # Update memory with resolved context
            followup_memory = {
                'last_resolved_follow_up': True,
                'last_resolved_query': resolved_query,
            }
            if resolved_day:
                followup_memory['last_day'] = resolved_day
            if resolved_subject:
                followup_memory['last_subject'] = resolved_subject
            if resolved_faculty:
                followup_memory['last_faculty'] = resolved_faculty
            if resolved_room:
                followup_memory['last_room'] = resolved_room
            _update_session_memory(session_id, followup_memory)

    # ── Centralized router: single point of query dispatching ──────────────────
    handler_type, response = route_query(original_query, class_name, schedule, 
                                         session_id=session_id, history=history)
    
    return response


# ── timetable → compact text ───────────────────────────────────────────────────

def _is_valid_class(val: str) -> bool:
    """Return True if a cell value is a real class, not a fragment."""
    if not val or val == '—':
        return False
    val = val.strip()
    if len(val) <= 3:
        return False
    if re.match(r'^B\d{3}[A-Z]?$', val):  # room-only like B116
        return False
    if re.match(r'^[A-Z]\s', val) and len(val) < 6:  # fragment like "L Ms."
        return False
    return True


def _timetable_to_text(tt: dict) -> str:
    """
    Build timetable text for LLM. Uses a DAY-FIRST format to prevent
    the LLM from mixing up columns.
    """
    class_name = tt.get('class', 'Unknown')
    schedule = tt.get('schedule', [])
    legend = tt.get('legend', {})

    now = datetime.now()
    today_idx = now.weekday()
    today_abbr = WEEKDAY_TO_ABBR[today_idx] if today_idx <= 4 else None
    today_name = DAY_FULL.get(today_abbr, 'Weekend') if today_abbr else 'Weekend'
    tomorrow_idx = today_idx + 1
    tomorrow_abbr = WEEKDAY_TO_ABBR[tomorrow_idx] if tomorrow_idx <= 4 else None
    tomorrow_name = DAY_FULL.get(tomorrow_abbr, 'Weekend') if tomorrow_abbr else 'Weekend'

    days = ["Mo", "Tu", "We", "Th", "Fr"]
    day_names = {"Mo": "MONDAY", "Tu": "TUESDAY", "We": "WEDNESDAY", "Th": "THURSDAY", "Fr": "FRIDAY"}

    lines = [
        f"Class: {class_name}",
        f"Today: {today_name} | Tomorrow: {tomorrow_name} | Current time: {now.strftime('%H:%M')}",
        "",
        "=" * 60,
        "TIMETABLE (organized by day — each section is ONE day only)",
        "=" * 60,
    ]

    # Output day-by-day to prevent column confusion
    for d in days:
        day_label = day_names[d]
        marker = " ← TODAY" if d == today_abbr else (" ← TOMORROW" if d == tomorrow_abbr else "")
        lines.append(f"\n{day_label}{marker}:")
        has_any = False
        for slot in schedule:
            val = slot.get('classes', {}).get(d, '').replace('\n', ' ').strip()
            if _is_valid_class(val):
                lines.append(f"  {slot['time']}: {val}")
                has_any = True
        if not has_any:
            lines.append("  (no classes)")

    # Authoritative counts
    lines.append("\n" + "=" * 60)
    lines.append("VERIFIED CLASS COUNT (do not recount, use these):")
    for d in days:
        valid = [s for s in schedule if _is_valid_class(s.get('classes', {}).get(d, ''))]
        lines.append(f"  {day_names[d]}: {len(valid)} classes")

    if legend:
        lines.append("\nLEGEND:")
        for k, v in legend.items():
            lines.append(f"  {k} = {v}")

    return "\n".join(lines)


# ── formatters ─────────────────────────────────────────────────────────────────

def _ask_branch(branches: list) -> str:
    opts = ''.join(f"<li>{b}</li>" for b in branches)
    return f"<div>What is your <strong>branch / programme</strong>?<ul style='margin:6px 0 0 16px'>{opts}</ul></div>"


def _ask_section(sections: list) -> str:
    opts = ''.join(f"<li>{s}</li>" for s in sections)
    return f"<div>Which <strong>section</strong> are you in?<ul style='margin:6px 0 0 16px'>{opts}</ul></div>"


def _format_full_schedule(class_name: str, schedule: list) -> str:
    return _render_full_table(class_name, schedule)


def _normalize_day(day_input: str = None) -> Optional[str]:
    """
    Unified day normalization. Converts any day specification to standard abbreviation.
    
    Handles:
      - Full names: "Monday" → "Mo", "Tuesday" → "Tu", etc.
      - Abbreviations: "Mo", "Tu", etc. → returned as-is
      - Keywords: "today", "tomorrow"
      - Typos/fuzzy matches (confidence >= 0.65)
    
    Returns: Day abbreviation ("Mo", "Tu", "We", "Th", "Fr") or None (weekends/invalid)
    """
    if not day_input:
        return None
    
    day_str = str(day_input).strip().lower()
    
    # Map of all day patterns to abbreviation
    day_map = {
        "monday": "Mo", "mon": "Mo", "mo": "Mo",
        "tuesday": "Tu", "tue": "Tu", "tu": "Tu",
        "wednesday": "We", "wed": "We", "we": "We",
        "thursday": "Th", "thu": "Th", "th": "Th",
        "friday": "Fr", "fri": "Fr", "fr": "Fr",
    }
    
    # Direct match
    if day_str in day_map:
        return day_map[day_str]
    
    # Substring match (e.g., "monday" in "timetable for monday")
    for pattern, abbr in day_map.items():
        if pattern in day_str:
            return abbr
    
    # Fuzzy match (typo tolerance)
    corrected_day, confidence, _ = _fuzzy_match_weekday(day_str)
    if corrected_day and confidence >= 0.65:
        corrected_lower = corrected_day.lower()
        if corrected_lower in day_map:
            return day_map[corrected_lower]
    
    return None


def _detect_day_from_msg(msg: str) -> Optional[str]:
    """
    Extract day abbreviation from message text using unified normalization.
    
    Handles "today", "tomorrow", and specific weekdays.
    Delegates to _normalize_day() for consistency.
    """
    if not msg:
        return None
    
    msg_lower = msg.lower()
    today_idx = datetime.now().weekday()
    
    # Handle special keywords first
    if 'tomorrow' in msg_lower:
        nxt = today_idx + 1
        if nxt <= 4:
            return WEEKDAY_TO_ABBR[nxt]
        return None
    
    if 'today' in msg_lower:
        if today_idx <= 4:
            return WEEKDAY_TO_ABBR[today_idx]
        return None
    
    # Try normalized day extraction from words
    words = msg_lower.split()
    for word in words:
        if len(word) >= 3:
            normalized = _normalize_day(word)
            if normalized:
                return normalized
    
    return None


def _render_full_table(class_name: str, schedule: list) -> str:
    """Render full week timetable as a styled HTML table."""
    days = ["Mo", "Tu", "We", "Th", "Fr"]
    day_labels = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

    header_cells = "".join(
        f"<th style='padding:8px 12px;background:#4c1d95;color:#e9d5ff;font-weight:600;white-space:nowrap'>{d}</th>"
        for d in day_labels
    )
    header = (
        f"<tr>"
        f"<th style='padding:8px 12px;background:#4c1d95;color:#e9d5ff;font-weight:600'>Time</th>"
        f"{header_cells}</tr>"
    )

    rows = []
    for s in schedule:
        cells = ""
        has_class = False
        for d in days:
            val = s.get('classes', {}).get(d, '')
            if val:
                has_class = True
                # Format: room + subject + faculty on separate lines
                parts = val.replace('\n', '<br>').strip()
                cells += f"<td style='padding:6px 10px;font-size:12px;color:#c4b5fd'>{parts}</td>"
            else:
                cells += "<td style='padding:6px 10px;color:#4b5563;text-align:center'>—</td>"
        row_bg = "background:#1e1b4b" if has_class else "background:#111827"
        rows.append(
            f"<tr style='{row_bg};border-bottom:1px solid #374151'>"
            f"<td style='padding:6px 10px;color:#9ca3af;white-space:nowrap;font-size:12px'>{s['time']}</td>"
            f"{cells}</tr>"
        )

    return (
        f"<div style='overflow-x:auto'>"
        f"<p style='color:#a78bfa;font-weight:600;margin-bottom:8px'>{class_name}</p>"
        f"<table style='border-collapse:collapse;width:100%;font-size:12px'>"
        f"<thead>{header}</thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        f"</table></div>"
    )


def _render_day_table(class_name: str, day: str, schedule: list) -> str:
    """Render a single day's timetable as a styled HTML table."""
    day_name = DAY_FULL.get(day, day)
    rows = []
    for s in schedule:
        val = s.get('classes', {}).get(day, '')
        if val:
            parts = val.replace('\n', '<br>').strip()
            rows.append(
                f"<tr style='background:#1e1b4b;border-bottom:1px solid #374151'>"
                f"<td style='padding:6px 12px;color:#9ca3af;white-space:nowrap;font-size:12px'>{s['time']}</td>"
                f"<td style='padding:6px 12px;color:#c4b5fd;font-size:12px'>{parts}</td>"
                f"</tr>"
            )

    if not rows:
        return f"<div>No classes on <strong>{day_name}</strong> for <strong>{class_name}</strong>.</div>"

    return (
        f"<div>"
        f"<p style='color:#a78bfa;font-weight:600;margin-bottom:8px'>{class_name} — {day_name}</p>"
        f"<table style='border-collapse:collapse;width:100%'>"
        f"<thead><tr>"
        f"<th style='padding:8px 12px;background:#4c1d95;color:#e9d5ff;text-align:left'>Time</th>"
        f"<th style='padding:8px 12px;background:#4c1d95;color:#e9d5ff;text-align:left'>Class</th>"
        f"</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        f"</table></div>"
    )
