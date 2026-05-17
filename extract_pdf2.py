import PyPDF2
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Extract remaining pages from FYP Report (from page 18 onwards)
report_path = None
scenario_path = None
for f in os.listdir(r'd:\FYP_Tranformer'):
    if f.endswith('.pdf'):
        if 'Adaptive' in f:
            report_path = os.path.join(r'd:\FYP_Tranformer', f)
        elif 'Scenario' in f:
            scenario_path = os.path.join(r'd:\FYP_Tranformer', f)

# Extract FYP Report remaining pages
if report_path:
    print("="*80)
    print("FYP REPORT - REMAINING PAGES")
    print("="*80)
    reader = PyPDF2.PdfReader(report_path)
    print(f"Total pages: {len(reader.pages)}")
    for i in range(17, len(reader.pages)):  # Start from page 18 (index 17)
        try:
            text = reader.pages[i].extract_text()
            if text:
                # Replace problematic characters
                text = text.encode('utf-8', errors='replace').decode('utf-8')
                print(f"\n--- PAGE {i+1} ---")
                print(text)
        except Exception as e:
            print(f"Error on page {i+1}: {e}")

# Extract ALL scenario pages
if scenario_path:
    print("\n" + "="*80)
    print("HRI SCENARIOS - ALL PAGES")
    print("="*80)
    reader = PyPDF2.PdfReader(scenario_path)
    print(f"Total pages: {len(reader.pages)}")
    for i in range(len(reader.pages)):
        try:
            text = reader.pages[i].extract_text()
            if text:
                text = text.encode('utf-8', errors='replace').decode('utf-8')
                print(f"\n--- PAGE {i+1} ---")
                print(text)
        except Exception as e:
            print(f"Error on page {i+1}: {e}")
