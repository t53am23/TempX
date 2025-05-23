import pandas as pd
import json
import re
from datetime import datetime
from google.cloud import translate_v2 as translate
from google.colab import files
from babel.dates import format_date
import dateparser
import os
import html

# Ask user to upload credentials and set up the translation client
def initialize_translator():
    print("Please upload your Google Cloud credentials JSON file.")
    uploaded = files.upload()
    credentials = next(iter(uploaded))
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = f"/content/{credentials}"
    return translate.Client()

translator = initialize_translator()

# Formatter for German localization tasks
class GermanFormatter:
    def __init__(self):
        # Map English month names to German using Babel's localization
        self.month_map = {
            datetime(2000, i, 1).strftime('%B'): format_date(datetime(2000, i, 1), format='MMMM', locale='de')
            for i in range(1, 13)
        }

# Attempt to fix messy JSON formatting that might break parsing
def clean_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        text = re.sub(r"(?<!\\)'", '"', text)
        text = re.sub(r",\s*([}\]])", r"\1", text)
        return json.loads(text)

# Wrap translation call and safely handle exceptions
def translate_text(text, lang_code):
    if not isinstance(text, str):
        text = str(text)
    try:
        decoded = html.unescape(text)
        result = translator.translate(decoded, target_language=lang_code)
        return result["translatedText"]
    except Exception as error:
        print(f"Translation issue: {error}")
        return text

# Translate and re-key common fields in structured data
def translate_json_fields(data, lang_code):
    key_swap = {
        "explanation": "Erklärung",
        "date": "Datum",
        "ordered_list": "Geordnete Liste",
        "Age": "Alter"
    }
    translated_output = {}
    for key, val in data.items():
        new_key = key_swap.get(key.lower(), key)
        if isinstance(val, str):
            translated_val = translate_text(val, lang_code)
        elif isinstance(val, list):
            translated_val = [translate_text(item, lang_code) if isinstance(item, str) else item for item in val]
        else:
            translated_val = val
        translated_output[new_key] = translated_val
    return translated_output

# Swap English date strings with localized German equivalents
def localize_dates_in_text(text):
    formatter = GermanFormatter()
    for eng_month, de_month in formatter.month_map.items():
        text = re.sub(rf"\b{eng_month}\b", de_month, text, flags=re.IGNORECASE)

    matches = re.findall(r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}", text)
    for match in matches:
        parsed = dateparser.parse(match)
        if parsed:
            german_style = format_date(parsed, format='long', locale='de')
            text = text.replace(match, german_style)
    return text

# Process each row by translating all relevant parts
def process_entry(row):
    q_raw = str(row.get("Question", ""))
    a_raw = str(row.get("Answer", ""))

    if "{" in q_raw:
        parsed_q = clean_json(q_raw)
        translated_q = json.dumps(translate_json_fields(parsed_q, "de"), ensure_ascii=False)
    else:
        translated_q = translate_text(q_raw, "de")

    if "{" in a_raw:
        parsed_a = clean_json(a_raw)
        translated_a = json.dumps(translate_json_fields(parsed_a, "de"), ensure_ascii=False)
    else:
        translated_a = translate_text(a_raw, "de")

    localized_q = localize_dates_in_text(translated_q)
    localized_a = localize_dates_in_text(translated_a)

    options_translated = {}
    for opt in ["Option A", "Option B", "Option C", "Option D"]:
        if opt in row and pd.notna(row[opt]):
            opt_text = translate_text(str(row[opt]), "de")
            options_translated[opt] = localize_dates_in_text(opt_text)

    return {"Question": localized_q, "Answer": localized_a, **options_translated}

# Main function to run the translation workflow
def start_translation_process():
    print("Upload the Excel file to be translated.")
    uploaded = files.upload()
    input_file = next(iter(uploaded))

    data = pd.read_excel(input_file)
    translated_data = []

    for _, row in data.iterrows():
        try:
            result = process_entry(row)
            translated_data.append(result)
        except Exception as err:
            print("Skipping a row due to error:", err)

    output_file = "Translated_German.xlsx"
    pd.DataFrame(translated_data).to_excel(output_file, index=False)
    files.download(output_file)
    print("Translation is complete. File saved as:", output_file)

# Launch when the script is run
if __name__ == "__main__":
    start_translation_process()
