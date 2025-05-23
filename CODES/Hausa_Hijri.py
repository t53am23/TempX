import pandas as pd
import re
from hijridate import Gregorian, Hijri  # Used for Hijri conversion
from random import randint
from datetime import datetime
from openpyxl import load_workbook

# --- FILE SETTINGS ---
INPUT_FILE = "Translated_Hausa_Short_ote.xlsx"
OUTPUT_FILE = "Normalized_Hijri_Dataset.xlsx"

# Columns that may contain dates
DATE_COLUMNS = ['Question', 'Option A', 'Option B', 'Option C', 'Option D',
                'Date', 'fact_context', 'context', 'Started_time', 'Closed_time']

# Hijri month names in a readable format
HIJRI_MONTHS = {
    "01": "Muharram", "02": "Safar", "03": "Rabi’ul-Awwal", "04": "Rabi’ul-Thani",
    "05": "Jumada al-Awwal", "06": "Jumada al-Thani", "07": "Rajab", "08": "Sha’ban",
    "09": "Ramadan", "10": "Shawwal", "11": "Dhul-Qa’ada", "12": "Dhul-Hijja"
}

# Convert a string into a Hijri date, if possible
def parse_date(date_str):
    if pd.isnull(date_str):
        return None
    text = str(date_str)

    # Check if it's already in Hijri format
    match = re.match(r'^(\d{1,2})-(\d{2})-(\d{4})', text)
    if match:
        day, month, year = map(int, match.groups())
        return Hijri(year, month, day, validate=False)

    # Try known Gregorian formats
    formats = ["%Y-%m-%d", "%d/%m/%Y", "%b %d, %Y", "%B %d, %Y"]
    for fmt in formats:
        try:
            parsed = datetime.strptime(text, fmt)
            return Gregorian(parsed.year, parsed.month, parsed.day).to_hijri()
        except:
            continue
    return None

# Shift Hijri dates based on tag: past, present, or future
def shift_hijri_date(hijri_date, tag):
    if tag == 'past':
        return Hijri(randint(1200, 1420), hijri_date.month, hijri_date.day)
    elif tag == 'future':
        return Hijri(randint(1451, 1509), hijri_date.month, hijri_date.day)
    return hijri_date

# Normalize any date expression within a text block
def normalize_text(text, shift_func):
    def convert_match(match):
        original = match.group(1)
        hijri = parse_date(original)
        if hijri:
            shifted = shift_func(hijri)
            return f"{shifted.day:02d} {HIJRI_MONTHS[f'{shifted.month:02d}']} {shifted.year}"
        return original

    # Recognize various date formats
    pattern = r'(\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4}\b)'
    return re.sub(pattern, convert_match, str(text))

# Apply date normalization to each row
def process_row(row):
    tags = str(row.get('Tags_list', '')).split(', ')
    tag = 'present'
    if 'past' in tags:
        tag = 'past'
    elif 'future' in tags:
        tag = 'future'

    for column in DATE_COLUMNS:
        if pd.notnull(row[column]):
            row[column] = normalize_text(row[column], lambda x: shift_hijri_date(x, tag))

    if tag not in tags:
        tags.append(tag)
    row['Tags_list'] = ', '.join(tags)
    return row

# --- MAIN WORKFLOW ---
# Load the Excel data
df = pd.read_excel(INPUT_FILE)

# Preserve original values for traceability
for col in ['Question', 'Date', 'fact_context', 'text_answers']:
    df[f'original_{col}'] = df[col]

# Normalize all rows
df = df.apply(process_row, axis=1)

# Save the updated file
df.to_excel(OUTPUT_FILE, index=False, engine='openpyxl')
print(f"Finished: {len(df)} rows processed. Output saved to '{OUTPUT_FILE}'.")
