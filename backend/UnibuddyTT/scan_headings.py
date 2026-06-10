import pdfplumber
import os

pdf_folder = r"C:\Users\saafi\OneDrive\Desktop\Time Tbales for Unibuddy"

# Keywords that suggest a section heading
keywords = ["year", "b.tech", "btech", "bca", "mca", "b.sc", "msc", "bcom", "section", "group", "sem", "semester"]

for pdf_file in sorted(os.listdir(pdf_folder)):
    if not pdf_file.endswith(".pdf"):
        continue
    print(f"\n{'='*60}")
    print(f"FILE: {pdf_file}")
    print('='*60)
    with pdfplumber.open(os.path.join(pdf_folder, pdf_file)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if not text:
                continue
            for line in text.split("\n"):
                stripped = line.strip()
                if not stripped:
                    continue
                lower = stripped.lower()
                if any(kw in lower for kw in keywords):
                    print(f"  p{i+1:03d}: {stripped}")
