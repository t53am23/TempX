import pandas as pd
import re
from hijri_converter import convert

# Define file paths and columns to be processed
input_file = "Translated_Hausa_Short_ote.xlsx"
output_file = "updated_hijri_dates.txt"
date_columns = [
    "Question", "Option A", "Option B", "Option C", "Option D",
    "Date", "fact_context", "context", "none_context"
]

# Define time ranges for different Hijri periods
hijri_ranges = {
    "historical": (100, 1342),
    "present": (1342, 1446),
    "future": (1450, 1508)
}

# Convert a date string from Gregorian to Hijri format
def convert_to_hijri(greg_date):
    try:
        d, m, y = map(int, greg_date.split('/'))
        hijri = convert.Gregorian(y, m, d).to_hijri()
        return f"{hijri.day}/{hijri.month}/{hijri.year}"
    except:
        return None

# Redistribute Hijri year based on row's position in the dataset
def shift_year(hijri_date, index, total):
    try:
        day, month, year = map(int, hijri_date.split('/'))
        if index < total // 3:
            new_year = hijri_ranges["historical"][0] + (index % (hijri_ranges["historical"][1] - hijri_ranges["historical"][0] + 1))
        elif index < 2 * total // 3:
            new_year = hijri_ranges["present"][0] + (index % (hijri_ranges["present"][1] - hijri_ranges["present"][0] + 1))
        else:
            new_year = hijri_ranges["future"][0] + (index % (hijri_ranges["future"][1] - hijri_ranges["future"][0] + 1))
        return f"{day}/{month}/{new_year}"
    except:
        return hijri_date

# Extract all date patterns (dd/mm/yyyy or dd-mm-yyyy) from text
def find_dates(text):
    return re.findall(r'\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4}\b', str(text))

# Load the Excel sheet into memory
df = pd.read_excel(input_file)

# First: convert any Gregorian date found to Hijri
for column in date_columns:
    df[column] = df[column].apply(
        lambda val: ' '.join([
            convert_to_hijri(date) if convert_to_hijri(date) else date
            for date in find_dates(val)
        ]) if isinstance(val, str) else val
    )

# Second: spread the Hijri dates into historical, present, and future categories
row_count = len(df)
for column in date_columns:
    df[column] = df.apply(
        lambda row: ' '.join([
            shift_year(date, row.name, row_count)
            for date in find_dates(row[column])
        ]) if isinstance(row[column], str) else row[column],
        axis=1
    )

# Save to output file
df.to_csv(output_file, sep="\t", index=False)
print(f"✅ Successfully saved updated file to: {output_file}")
