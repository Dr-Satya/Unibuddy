from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path
import io
import re
import pdfplumber

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

from src.api_adapter import get_reply
from src.timetable_store import save_timetable

app = FastAPI(title="UniBuddy API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


DAYS = ["Mo", "Tu", "We", "Th", "Fr"]
TIME_SLOTS = [
    "9:10 - 10:00", "10:00 - 10:50", "10:50 - 11:40",
    "11:40 - 12:30", "12:30 - 13:20", "13:20 - 14:10",
    "14:10 - 15:00", "15:00 - 15:50", "15:50 - 16:30",
]

# Words to never include in schedule cells
SKIP_TOKENS = {
    'aSc', 'Timetables', 'generated', 'Short', 'Name',
    'G', 'D', 'Goenka', 'University', 'Sohna', 'Road',
    'Gurgaon', 'Haryana', 'School', 'Of', 'Engineering',
    'Sciences', 'and', 'the', 'Mo', 'Tu', 'We', 'Th', 'Fr',
}


def parse_timetable_page(page) -> dict:
    """
    Parse one page of a GD Goenka timetable PDF using word bounding boxes.
    Correctly handles 2-hour labs (merged/spanning cells).
    """
    text = page.extract_text() or ""
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    # ── Class name ────────────────────────────────────────────────────────────
    class_name = ""
    for i, line in enumerate(lines):
        if "School Of Engineering" in line and i + 1 < len(lines):
            class_name = lines[i + 1]
            break

    # ── Extract all words with bounding boxes ─────────────────────────────────
    words = page.extract_words(x_tolerance=3, y_tolerance=3, keep_blank_chars=False)
    if not words:
        return _empty_page(class_name, text)

    # ── Locate day label words (Mo/Tu/We/Th/Fr) ───────────────────────────────
    day_words = [w for w in words if w['text'] in DAYS]
    if not day_words:
        return _empty_page(class_name, text)

    # Use first occurrence of each day
    day_y = {}
    for w in day_words:
        if w['text'] not in day_y:
            day_y[w['text']] = {'top': w['top'], 'bottom': w['bottom'], 'x1': w['x1']}

    sorted_days = sorted(day_y.items(), key=lambda x: x[1]['top'])
    first_day_top = sorted_days[0][1]['top']
    last_day_top = sorted_days[-1][1]['top']

    # Row height estimate
    if len(sorted_days) > 1:
        row_h = (last_day_top - first_day_top) / (len(sorted_days) - 1)
    else:
        row_h = 50

    # Grid bottom: just below last day row
    grid_bottom = last_day_top + row_h * 1.05

    # Day row boundaries (y0, y1)
    day_bounds = {}
    for i, (day, info) in enumerate(sorted_days):
        y0 = info['top'] - row_h * 0.45
        y1 = info['top'] + row_h * 0.55
        day_bounds[day] = (y0, y1)

    # ── Locate slot header numbers (1-9) ──────────────────────────────────────
    slot_headers = {}  # slot_idx (0-based) -> x_center
    for w in words:
        if w['text'].isdigit() and 1 <= int(w['text']) <= 9:
            if w['top'] < first_day_top - 5:  # must be above first day row
                idx = int(w['text']) - 1
                if idx not in slot_headers:
                    slot_headers[idx] = (w['x0'] + w['x1']) / 2

    if len(slot_headers) < 3:
        return _empty_page(class_name, text)

    sorted_slots = sorted(slot_headers.items())  # [(idx, cx), ...]

    # Build column x-boundaries
    col_bounds = []  # [(slot_idx, x0, x1), ...]
    for i, (idx, cx) in enumerate(sorted_slots):
        x0 = 0 if i == 0 else (sorted_slots[i-1][1] + cx) / 2
        x1 = page.width if i == len(sorted_slots)-1 else (cx + sorted_slots[i+1][1]) / 2
        col_bounds.append((idx, x0, x1))

    # Day label column right edge
    day_col_right = max(info['x1'] for _, info in day_y.items()) + 4

    # ── Map each word to (day, slot) ──────────────────────────────────────────
    # cell_data[(day, slot)] = list of word texts
    cell_data: dict = {}

    for w in words:
        # Skip words outside the grid
        if w['top'] < first_day_top - 8:
            continue
        if w['top'] > grid_bottom:
            continue
        if w['x0'] < day_col_right:
            continue
        if w['text'] in SKIP_TOKENS:
            continue
        # Skip pure numbers that are slot headers
        if w['text'].isdigit():
            continue

        # Find day row
        day = None
        for d, (y0, y1) in day_bounds.items():
            if y0 <= w['top'] <= y1 or y0 <= w['bottom'] <= y1:
                day = d
                break
        if not day:
            continue

        # Find all slot columns this word overlaps (handles spanning cells)
        for slot_idx, x0, x1 in col_bounds:
            # Word overlaps column if their x-ranges intersect
            if w['x1'] > x0 + 1 and w['x0'] < x1 - 1:
                key = (day, slot_idx)
                if key not in cell_data:
                    cell_data[key] = []
                if w['text'] not in cell_data[key]:
                    cell_data[key].append(w['text'])

    # ── Build schedule ────────────────────────────────────────────────────────
    schedule = []
    for slot_idx, time in enumerate(TIME_SLOTS):
        entry = {"slot": slot_idx + 1, "time": time, "classes": {}}
        for day in DAYS:
            tokens = cell_data.get((day, slot_idx), [])
            if tokens:
                entry["classes"][day] = " ".join(tokens)
        schedule.append(entry)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend = _extract_legend(lines)

    return {
        "class": class_name,
        "schedule": schedule,
        "legend": legend,
        "raw_text": text,
    }


def _empty_page(class_name: str, text: str) -> dict:
    return {
        "class": class_name,
        "schedule": [{"slot": i+1, "time": t, "classes": {}} for i, t in enumerate(TIME_SLOTS)],
        "legend": _extract_legend([l.strip() for l in text.splitlines() if l.strip()]),
        "raw_text": text,
    }


def _extract_legend(lines: list) -> dict:
    legend = {}
    for line in lines:
        m = re.match(r'^([A-Z]{2,6}[0-9]*(?:/[A-Z0-9]+)?)\s+(.+)$', line)
        if m:
            code, name = m.group(1), m.group(2).strip()
            if code not in DAYS and not code.isdigit() and len(code) >= 2:
                legend[code] = name
    return legend


@app.post('/timetable')
async def upload_timetable(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail='Only PDF files are allowed.')
    try:
        contents = await file.read()
        pdf = pdfplumber.open(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f'Invalid PDF file: {exc}')

    result = []
    with pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            parsed = parse_timetable_page(page)
            parsed["page"] = page_num
            result.append(parsed)

    parsed_result = {"filename": file.filename, "total_pages": len(result), "timetables": result}
    saved_classes = save_timetable(parsed_result)

    return {
        **parsed_result,
        "saved_classes": saved_classes,
        "message": f"Saved {len(saved_classes)} class timetable(s) to chatbot knowledge base.",
    }


@app.post('/chat')
async def chat(req: ChatRequest):
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail='Empty message')
    result = get_reply(req.message, session_id=req.session_id)
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=9000, reload=False)
