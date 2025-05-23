import pandas as pd
import re
from datetime import datetime
import string

# List of common date formats we want to support
DATE_FORMATS = [
    "%Y-%m-%d", "%m-%d-%Y", "%d/%m/%Y", "%b %d, %Y",
    "%B %d, %Y", "%d-%b-%Y", "%d-%B-%Y",
    "%d %B %Y", "%Y-%b-%d", "%m/%d/%Y",
    "%d %b, %Y", "%d %B, %Y", "%d %b %Y", "%d-%m-%Y"
]

# This is the format all dates will be converted to
OUTPUT_FORMAT = "%Y-%m-%d"

# Function to clean up and convert a date string to a standard format
def normalize_date(date_str):
    if not isinstance(date_str, str):
        return date_str

    # Clean up spacing, punctuation, and suffixes (like "th", "st", etc.)
    cleaned = date_str.strip()
    cleaned = cleaned.replace(".", "")
    cleaned = re.sub(r'(\d{1,2})(st|nd|rd|th)', r'\1', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r'\s*,', ',', cleaned)
    cleaned = re.sub(r',', ', ', cleaned)

    # Try matching the cleaned string to known formats
    for fmt in DATE_FORMATS:
        try:
            parsed = datetime.strptime(cleaned, fmt)
            return parsed.strftime(OUTPUT_FORMAT)
        except ValueError:
            continue

    # If we can't parse it using known formats, try basic patterns
    if re.fullmatch(r'\d{4}-\d{1,2}-\d{1,2}', cleaned):
        year, month, day = cleaned.split('-')
    elif re.fullmatch(r'\d{1,2}/\d{1,2}/\d{4}', cleaned):
        month, day, year = cleaned.split('/')
    elif re.fullmatch(r'\d{1,2}-\d{1,2}-\d{4}', cleaned):
        day, month, year = cleaned.split('-')
    else:
        return date_str

    # Fix impossible dates by clamping the day to max 28
    try:
        parsed = datetime(int(year), int(month), min(int(day), 28))
        return parsed.strftime(OUTPUT_FORMAT)
    except:
        try:
            month_name = datetime(1900, int(month), 1).strftime("%B")
        except:
            month_name = "Unknown"
        return f"{month_name} {day.zfill(2)}, {year}"

# This looks for date-like text in a string and converts it
def replace_dates_in_text(cell):
    if isinstance(cell, datetime):
        return cell.strftime(OUTPUT_FORMAT)
    elif not isinstance(cell, str):
        return cell

    # Pattern that matches different ways dates might be written
    date_pattern = r"""
        (
            \d{4}-\d{1,2}-\d{1,2}|
            \d{1,2}/\d{1,2}/\d{4}|
            [A-Za-z]{3}\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{4}|
            [A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{4}|
            \d{1,2}-[A-Za-z]{3,9}-\d{4}|
            \d{1,2}\s+[A-Za-z]+\s*,?\s*\d{4}|
            \d{4}-[A-Za-z]{3}-\d{2}|
            \d{1,2}-\d{1,2}-\d{4}
        )
        (\S*)
    """

    # Helper to replace any found match with the cleaned version
    def replace_match(match):
        date = normalize_date(match.group(1))
        trailing = match.group(2)
        if trailing:
            if trailing[0] in string.punctuation:
                return date + trailing
            return date + ' ' + trailing
        return date

    return re.sub(date_pattern, replace_match, cell, flags=re.VERBOSE)

# Apply date normalization to a whole CSV file
def normalize_dates_in_file(input_file, output_file):
    data = pd.read_csv(input_file)
    for column in data.columns:
        data[column] = data[column].apply(replace_dates_in_text)
    data.to_excel(output_file, index=False)
    print(f"Dates normalized and saved to {output_file}")

# Run the function with your file
input_file = "/content/Normalized_German_ISO.csv"
output_file = "Normalized_German_ISO3.xlsx"
normalize_dates_in_file(input_file, output_file)
