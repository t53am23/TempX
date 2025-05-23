import pandas as pd
import json
import re
from datetime import datetime
import time
import random
import html
from google.cloud import translate_v2 as translate
from google.colab import files
from babel.dates import format_date
from babel.numbers import format_decimal
import dateparser
import os

# ========================== CONFIGURATION SETTINGS ==========================
print("1. Upload Google Cloud Service Account JSON:")
service_account = files.upload()
service_account_name = next(iter(service_account))
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = f"/content/{service_account_name}"
translator = translate.Client()

class LocalizationConfig:
    """Handles date, number, and term localization for Chinese."""
    def __init__(self, lang='zh'):
        self.lang = lang
        self.numeral_map = str.maketrans("0123456789", "〇一二三四五六七八九") if lang == "zh" else {}
        self.month_map = {
            "January": "一月", "February": "二月", "March": "三月", "April": "四月",
            "May": "五月", "June": "六月", "July": "七月", "August": "八月",
            "September": "九月", "October": "十月", "November": "十一月", "December": "十二月"
        } if lang == "zh" else {}
        self.term_map = {
            "Explanation": "解释", "Date": "日期", "Day": "天",
            "time": "时间", "unordered_list": "无序列表"
        } if lang == "zh" else {}

# ========================== JSON PARSING FUNCTION ==========================
def safe_json_parse(text):
    """Safely parses JSON, correcting common formatting errors."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            fixed_text = re.sub(r"(?<!\\)'", '"', text)
            fixed_text = re.sub(r',\s*([}\]])', r'\1', fixed_text)  # Remove trailing commas
            return json.loads(fixed_text)
        except:
            return text  # Return original text if still not valid JSON

# ========================== JSON TRANSLATION ==========================
def translate_json(json_data, target_language):
    """Recursively translates JSON keys and values while ensuring full translation integrity."""
    translated = {}

    json_key_fixes = {
        "explanation": "解释",
        "date": "日期",
        "ordered_list": "有序列表",
        "unordered_list": "无序列表",
        "Age": "年龄"
    }

    for key, value in json_data.items():
        corrected_key = translate_text(key, target_language).strip()
        translated_key = json_key_fixes.get(corrected_key.lower(), corrected_key)

        # Ensure full step-by-step translation
        if isinstance(value, str):
            fixed_value = translate_text(value, target_language).strip()
            fixed_value = re.sub(r"\b(Yyyy|YYYY)-MM-DD\b", "yyyy-mm-dd", fixed_value, flags=re.IGNORECASE)
            translated_value = fixed_value
        elif isinstance(value, list):
            translated_value = [translate_text(item, target_language) if isinstance(item, str) else item for item in value]
        else:
            translated_value = value  # Keep numbers unchanged

        translated[translated_key] = translated_value

    return translated

# ========================== TEXT TRANSLATION ==========================
def translate_text(text, target_language="de"):
    """Handles text translation using Google Cloud API."""
    try:
        # Convert any non-string input to string
        if not isinstance(text, str):
            text = str(text)
        result = translator.translate(text, target_language=target_language)
        return result["translatedText"]
    except Exception as e:
        print(f" Translation Error: {str(e)}")
        return text  # Return original text if translation fails


# ========================== DATE LOCALIZATION ==========================
def localize_text(text, config):
    """Applies full date and number localization, ensuring formats are consistent."""
    for eng, trans in config.month_map.items():
        text = re.sub(rf'\b{eng}\b', trans, text, flags=re.IGNORECASE)

    text = text.translate(config.numeral_map) if config.numeral_map else text

    date_matches = re.findall(r'\b\d{1,4}[-/]\d{1,2}[-/]\d{1,4}\b', text)
    for match in date_matches:
        dt = dateparser.parse(match)
        if dt:
            text = text.replace(match, format_date(dt, format='long', locale=config.lang))

    for eng, trans in config.term_map.items():
        text = text.replace(eng, trans)

    return text

# ========================== MAIN TRANSLATION PIPELINE ==========================
def process_translation(row, target_language):
    config = LocalizationConfig(target_language)

    # Ensure datetime is correctly referenced
    from datetime import datetime

    # Convert question and answer to string in case they are datetime
    question = str(row["Question"]) if isinstance(row["Question"], datetime) else row["Question"]
    answer = str(row["Answer"]) if isinstance(row["Answer"], datetime) else row["Answer"]

    # Handle optional columns dynamically and convert datetime if necessary
    options = {
        col: str(row[col]) if (col in row and isinstance(row[col], datetime)) else row[col]
        for col in ["Option A", "Option B", "Option C", "Option D"]
        if col in row and pd.notna(row[col])
    }

    # Ensure proper JSON parsing
    question_json = safe_json_parse(question) if isinstance(question, str) and question.startswith("{") else question
    answer_json = safe_json_parse(answer) if isinstance(answer, str) and answer.startswith("{") else answer

    # Translate the main question and answer
    translated_question = translate_text(question, target_language)
    translated_answer = translate_text(answer, target_language)

    # Translate answer choices dynamically
    translated_options = {
        col: translate_text(opt, target_language) for col, opt in options.items()
    }

    if isinstance(question_json, dict):
        translated_question = json.dumps(translate_json(question_json, target_language), ensure_ascii=False)
    if isinstance(answer_json, dict):
        translated_answer = json.dumps(translate_json(answer_json, target_language), ensure_ascii=False)

    final_question = localize_text(translated_question, config)
    final_answer = localize_text(translated_answer, config)
    final_options = {col: localize_text(opt, config) for col, opt in translated_options.items()}

    return {
        "Translated_Question": final_question,
        "Translated_Answer": final_answer,
        **final_options
    }

# ========================== FILE PROCESSING FUNCTION ==========================
def process_file(target_language="zh"):
    """Handles file upload, processing, and saving the translated results."""
    print(" Upload Excel file:")
    uploaded = files.upload()
    file_name = next(iter(uploaded))

    df = pd.read_excel(file_name)

    results = []
    for idx, row in df.iterrows():
        try:
            translated_data = process_translation(row, target_language)
            results.append(translated_data)
        except Exception as e:
            print(f" Error processing row {idx}: {str(e)}")

    # Save file
    output_file = f"Translated_{target_language}.xlsx"
    pd.DataFrame(results).to_excel(output_file, index=False)
    files.download(output_file)
    print(f" Translation completed! File saved as {output_file}")

# ========================== RUN SCRIPT ==========================
if __name__ == "__main__":
    process_file(target_language="zh")