import PyPDF2
import os
import sys

# Extract FYP Report
report_files = [f for f in os.listdir(r'd:\FYP_Tranformer') if f.endswith('.pdf')]
print("Found PDF files:", report_files)

for fname in report_files:
    fpath = os.path.join(r'd:\FYP_Tranformer', fname)
    print(f"\n{'='*80}")
    print(f"FILE: {fname}")
    print(f"{'='*80}")
    try:
        reader = PyPDF2.PdfReader(fpath)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                print(f"\n--- PAGE {i+1} ---")
                print(text)
    except Exception as e:
        print(f"Error reading {fname}: {e}")
