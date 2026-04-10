import camelot
import pdfplumber
import json
import re
import os
import openpyxl
import warnings
import logging

# Suppress camelot/ghostscript cleanup noise on Windows
warnings.filterwarnings("ignore")
logging.getLogger("camelot").setLevel(logging.ERROR)

pdf_folder = r"C:\Users\saafi\OneDrive\Desktop\Time Tbales for Unibuddy"

section_map = {}
timetable = []

time_slots = {
    1: "9:10-10:00",
    2: "10:00-10:50",
    3: "10:50-11:40",
    4: "11:40-12:30",
    5: "12:30-13:20",
    6: "13:20-14:10",
    7: "14:10-15:00",
    8: "15:00-15:50",
    9: "15:50-16:30"
}

valid_days = ["Mo", "Tu", "We", "Th", "Fr"]


def parse_section_heading(line):
    """
    Handles all observed heading formats:
      3rd Year B.Tech CSE A (AIML A)
      3rd Year B.Tech CSE AIML B
      3rd Year B.Tech CS
      3rd Year B.Tech CSE Core
      3rd Year B.Tech Data Science
      2nd Year B.Tech ECE
      2nd Year B.Tech CE (SI)
      2nd Year B.Tech Fire and Safety
      2nd Year BCA Sec DA / Sec ML
      2nd Year MCA
      3rd Year BCA DA / BCA ML
      3rd Year Diploma CSE
      1st Year B.Sc. (H) F.Sc.
      1st year B.Sc. Biotechnology / Microbiology
      2nd year M.Sc. F.Sc.
      4th Year B.Sc. B.Ed.
      1st Year B.Tech. CSE A1 / B2 / C
      1st Year B.Tech ECE (IoT/BME/VLSI)
      1st B.Tech. Aerospace
      4th Year B.Tech DS
      4th Year B.Tech CSE Core A / Core B
      4th Year B.Tech CSE AIML Sec T4
      4th Year B.Tech CSE CS Sec T4
      4th year B.tech-ME/MEX
      4th Year Int. B.Tech. CSE MBA
      5th Year Int. B.Tech. CSE MBA
      1st Year M.Tech. Environmental
      3rd Year CVT / Nutrition and Health
      1st Year MBAHM / BBAHM / BPT / B. Optometry
    """
    raw = line.strip().rstrip(".")

    # Match year at start (e.g. "3rd Year", "1st")
    year_match = re.match(r"^(\d+)\s*(?:ST|ND|RD|TH)?\s+YEAR\b", raw, re.IGNORECASE)
    if not year_match:
        # "1st B.Tech. Aerospace" — year without "Year" keyword
        year_match = re.match(r"^(\d+)\s*(?:ST|ND|RD|TH)?\s+(?=B\.?\s*TECH)", raw, re.IGNORECASE)
        if not year_match:
            return None

    year = year_match.group(1)
    rest = raw[year_match.end():].strip()

    # Degree patterns — order matters (longer/more specific first)
    degree_patterns = [
        (r"INT\.?\s*B\.?\s*TECH\.?\s*M\.?\s*TECH", "INT_BTECH_MTECH"),
        (r"INT\.?\s*B\.?\s*TECH", "INT_BTECH"),
        (r"M\.?\s*TECH", "MTECH"),
        (r"B\.?\s*TECH", "BTECH"),
        (r"B\.?\s*SC\.?\s*\(H\)", "BSC_H"),
        (r"B\.?\s*SC", "BSC"),
        (r"M\.?\s*SC", "MSC"),
        (r"BCA", "BCA"),
        (r"MCA", "MCA"),
        (r"DIPLOMA", "DIPLOMA"),
        (r"BCOM", "BCOM"),
        (r"BPT", "BPT"),
        (r"B\.?\s*OPTOMETRY", "BOPTOMETRY"),
        (r"MBAHM", "MBAHM"),
        (r"BBAHM", "BBAHM"),
        (r"CVT", "CVT"),
        (r"AEROSPACE", "AEROSPACE"),
        (r"NUTRITION\s+AND\s+HEALTH", "NUTRITION_HEALTH"),
    ]

    degree = ""
    for pattern, label in degree_patterns:
        m = re.match(pattern, rest, re.IGNORECASE)
        if m:
            degree = label
            rest = rest[m.end():].strip()
            break

    if not degree:
        # fallback: first word
        parts = rest.split()
        degree = parts[0].upper() if parts else "UNKNOWN"
        rest = " ".join(parts[1:])

    # Clean up rest into a branch/spec key
    branch = rest.strip().lstrip("-").strip()
    branch = re.sub(r"\s+", "_", branch)
    branch = re.sub(r"[^A-Z0-9_()]", "", branch.upper())
    branch = branch.strip("_")

    return f"{degree}|{year}|{branch}" if branch else f"{degree}|{year}"


# Load course and faculty mappings from Excel
def load_mapping(filepath):
    mapping = {}
    wb = openpyxl.load_workbook(filepath)
    ws = wb.active
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[0] and row[1]:
            name = str(row[0]).strip()
            short = str(row[1]).strip()
            mapping[short] = name
    return mapping

course_map = load_mapping(os.path.join(pdf_folder, "Subjects-Code-CSE-Even-Sem-2026.xlsx"))
faculty_map = load_mapping(os.path.join(pdf_folder, "Teacher-Abbrevation.xlsx"))
print(f"Loaded {len(course_map)} course codes, {len(faculty_map)} faculty abbreviations")

pdf_files = sorted(f for f in os.listdir(pdf_folder) if f.endswith(".pdf"))
print(f"Found {len(pdf_files)} PDF(s): {pdf_files}")

for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_folder, pdf_file)
    print(f"\nProcessing: {pdf_file}")

    # Detect sections — only check first 5 lines of each page
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if not text:
                continue
            for line in text.split("\n")[:5]:
                key = parse_section_heading(line.strip())
                if key:
                    entry = {"file": pdf_file, "page": i + 1}
                    existing = section_map.setdefault(key, [])
                    if entry not in existing:
                        existing.append(entry)
                    break  # one heading per page

    # Extract tables
    try:
        tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
        print(f"  Tables found: {len(tables)}")
    except Exception as e:
        print(f"  Warning: {e}")
        continue

    for table in tables:
        df = table.df
        page = table.page
        for i in range(len(df)):
            row = df.iloc[i]
            day = str(row[0]).strip()
            if day not in valid_days:
                continue
            for col in range(1, len(row)):
                cell = str(row[col]).strip()
                parts = cell.split("\n") if cell and cell.lower() != "nan" else []
                if len(parts) < 2:
                    timetable.append({
                        "day": day,
                        "time": time_slots.get(col, ""),
                        "room": "FREE",
                        "course": "FREE",
                        "course_code": "",
                        "faculty": "FREE",
                        "faculty_short": "",
                        "page": page,
                        "source": pdf_file
                    })
                else:
                    course_code = parts[1]
                    faculty_short = parts[2].strip() if len(parts) >= 3 else ""
                    timetable.append({
                        "day": day,
                        "time": time_slots.get(col, ""),
                        "room": parts[0],
                        "course": course_map.get(course_code, course_code),
                        "course_code": course_code,
                        "faculty": faculty_map.get(faculty_short, faculty_short),
                        "faculty_short": faculty_short,
                        "page": page,
                        "source": pdf_file
                    })

# Save
with open("timetable.json", "w") as f:
    json.dump(timetable, f, indent=2)

with open("sections.json", "w") as f:
    json.dump(section_map, f, indent=2)

print("\n✅ Preprocessing done!")
print(f"   {len(timetable)} timetable entries")
print(f"   {len(section_map)} sections detected")
