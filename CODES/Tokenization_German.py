import pandas as pd 
import numpy as np
import re
from datetime import datetime
from collections import Counter
from transformers import AutoTokenizer

# Load dataset
df = pd.read_csv("Normalized_German_Merged_Dataset.csv") 
df.columns = df.columns.str.strip() 



# Clean the 'Date' column
df['Date'] = (
    df['Date']
    .astype(str)
    .str.replace(r"^b'|^b\"|\'$|\"$", "", regex=True)  # Removes byte string artifacts
    .str.strip()
    .replace(["nan", "NaN", "None", ""], np.nan)       # Normalizes missing entries
)

def extract_date_from_question(question):
    patterns = [
        r"\b\d{2}/\d{2}/\d{4}\b",                      # DD/MM/YYYY or MM/DD/YYYY
        r"\b\d{4}-\d{2}-\d{2}\b",                      # YYYY-MM-DD (ISO)
        r"\b\d{2}-\d{2}-\d{4}\b",                      # DD-MM-YYYY
        r"\b\d{4}/\d{2}/\d{2}\b",                      # YYYY/MM/DD
        r"\b\d{4}\.\d{2}\.\d{2}\b",                    # YYYY.MM.DD
        r"\b\d{8}\b",                                  # YYYYMMDD
        r"\b\d{2}\.\d{2}\.\d{2}\b",                    # DD.MM.YY
        r"\b\d{2}\.\d{2}\.\d{4}\b",                    # DD.MM.YYYY
        r"\b\d{1,2}\. (?:Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember) \d{4}\b",  # DD. Month YYYY
        r"\b(?:Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember) \d{1,2},? \d{4}\b", # Month DD, YYYY
        r"\b\d{1,2} (?:Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember) \d{4}\b",   # DD Month YYYY
        r"^(?:Montag|Dienstag|Mittwoch|Donnerstag|Freitag|Samstag|Sonntag), der \d{1,2}\. (?:Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember) \d{4}$",  # Full formal German
    ]
    for pattern in patterns:
        match = re.search(pattern, question)
        if match:
            return match.group()
    return np.nan


df['Date'] = df.apply(
    lambda row: row['Date'] if pd.notna(row['Date']) else extract_date_from_question(str(row.get('Question', ''))),
    axis=1
)



# ---------------- Utilities ---------------- #

def baseline_tokenizer(date_str, format_type):
    format_map = {
        'YYYY-MM-DD': '%Y-%m-%d', 'YYYY/MM/DD': '%Y/%m/%d',
        'YYYY.MM.DD': '%Y.%m.%d', 'DD-MM-YYYY': '%d-%m-%Y',
        'DD/MM/YYYY': '%d/%m/%Y', 'MM/DD/YYYY': '%m/%d/%Y',
        'YYYYMMDD': '%Y%m%d', 'MMDDYYYY': '%m%d%Y', 'DDMMYYYY': '%d%m%Y',
        'Month DD, YYYY': '%B %d, %Y', 'DD Month YYYY': '%d %B %Y',
        'Month DD YYYY': '%B %d %Y', 'YYYY/DDD': '%Y/%j',
        'DDD/YYYY': '%j/%Y', 'YYYYDDD': '%Y%j', 'DDDYYYY': '%j%Y',
        '%A, der %-d. %B %Y': '%A, der %-d. %B %Y',
        'DD.MM.YY': '%d.%m.%y', 'DD.MM.YYYY': '%d.%m.%Y'  # Added formats
    }

    if format_type not in format_map:
        return [date_str]

    try:
        date_obj = datetime.strptime(date_str, format_map[format_type])

        # Handle "%A, der %-d. %B %Y" (e.g., "6. August 2023")
        if format_type == '%A, der %-d. %B %Y':
            parts = date_str.split(', der ')
            weekday = parts[0]
            day_part = parts[1].split('. ')[0]  # Directly extract "6" from input
            month_year = parts[1].split('. ')[1]
            month = month_year.split(' ')[0]
            year = month_year.split(' ')[1]
            return [weekday, ',', 'der', day_part, '.', month, year]

        # Handle "DD.MM.YYYY" or "DD.MM.YY" (e.g., "10.01.57")
        elif format_type in ['DD.MM.YYYY', 'DD.MM.YY']:
            day, month, year = date_str.split('.')  # Preserve original formatting
            return [day, '.', month, '.', year]

        # Other cases remain unchanged
        elif format_type in ['YYYY-MM-DD', 'YYYY/MM/DD', 'YYYY.MM.DD']:
            return [date_obj.strftime('%Y'), date_str[4], date_obj.strftime('%m'), date_str[7], date_obj.strftime('%d')]
        elif format_type in ['DD-MM-YYYY', 'DD/MM/YYYY']:
            return [date_obj.strftime('%d'), date_str[2], date_obj.strftime('%m'), date_str[5], date_obj.strftime('%Y')]
        elif format_type in ['MM/DD/YYYY']:
            return [date_obj.strftime('%m'), date_str[2], date_obj.strftime('%d'), date_str[5], date_obj.strftime('%Y')]
        elif format_type == 'YYYYMMDD':
            return [date_obj.strftime('%Y'), date_obj.strftime('%m'), date_obj.strftime('%d')]
       # elif format_type in ['DD.MM.YYYY', 'DD.MM.YY']:  # Handle new formats
           # day = date_obj.strftime('%d')
           # month = date_obj.strftime('%m')
           # year_fmt = '%Y' if format_type == 'DD.MM.YYYY' else '%y'
           # year = date_obj.strftime(year_fmt)
           # return [day, '.', month, '.', year]
        else:
            return [date_str]

    except Exception:
        return [date_str]


LANGUAGE_CONFIG = {
    'Hausa': {
        'time_patterns': [
            r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:na safe|na yamma|na dare|AM|PM|\(24hr\))\b',  # Matches "1 na safe" [[1]]
            r'\b\d{1,2}\s*(na safe|na yamma|na dare)\b',  # Matches "1 na safe" without colon [[2]]
            r'\b\d{1,2} ga (?:Janairu|Fabrairu|Maris|Afrilu|Mayu|Yuni|Yuli|Agusta|Satumba|Oktoba|Nuwamba|Disamba)\b',  # Detects "ga Month" [[3]]
            r'\b\d{1,2}:\d{2}(?::\d{2})?\s*\(?24hr\)?\b'  # Explicit 24-hour format [[4]]
        ],
        'timezone_patterns': [
            r'\b[A-Za-z]+/[A-Za-z_]+\b',  # Matches "Amurka/Mountain" [[5]]
            r'\bUTC[+-]?\d{2}:?\d{2}\b',  # Matches "UTC+8" [[6]]
            r'\b(WAT|EAT|CAT)\b'          # Hausa-specific timezone abbreviations [[7]]
        ]
    },
    'German': {
        'time_patterns': [
            r'\b\d{1,2}[:\.]\d{2}\s*Uhr\b',  # German 18:00 Uhr [[8]]
            r'\b\d{1,2}\s*(Uhr|h)\b',        # Simplified time [[9]]
            r'\b\d{1,2}:\d{2}(?::\d{2})?\s*\(?24Uhr\)?\b'  # 24-hour format [[10]]
        ],
        'timezone_patterns': [
            r'\b[A-Za-z]+/[A-Za-z_]+\b',  # Matches "Europe/Berlin" [[11]]
            r'\bMEZ|MESZ\b',              # German timezone abbreviations [[12]]
            r'\bUTC[+-]?\d{2}:?\d{2}\b'   # UTC+01:00 [[13]]
        ]
    },
    'Chinese': {
        'time_patterns': [
            r'(上午|下午|晚间)\s*\d{1,2}(点\d{1,2}(分\d{1,2})?)?',  # 上午3点15分 [[14]]
            r'\b\d{1,2}:\d{2}(?::\d{2})?\s*\(?24小时\)?\b'         # 18:00 (24hr) [[15]]
        ],
        'timezone_patterns': [
            r'\b[A-Za-z]+/[A-Za-z_]+\b',  # Matches "Asia/Shanghai" [[16]]
            r'\b北京时间|中国标准时间\b',  # 北京时间 [[17]]
            r'\bUTC\+?8\b'               # UTC+8 [[18]]
        ]
    },
    'English': {
        'time_patterns': [
            r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|24hr)\b',  # 6:30 PM [[1]]
            r'\b\d{1,2}\s*(?:AM|PM)\b'                          # 6 PM [[2]]
        ],
        'timezone_patterns': [
            r'\b[A-Za-z]+/[A-Za-z_]+\b',  # Matches "US/Hawaii" [[5]]
            r'\bUTC|GMT|EST|PST|CET\b',   # Timezone abbreviations [[19]]
            r'\bUTC[+-]?\d{2}:?\d{2}\b'   # UTC+8 [[20]]
        ]
    }
}


def detect_language(question):
    text = question.lower()
    if any(re.search(p, text, flags=re.IGNORECASE) for p in LANGUAGE_CONFIG['Hausa']['time_patterns']):
        return 'Hausa'
    if any(re.search(p, text, flags=re.IGNORECASE) for p in LANGUAGE_CONFIG['German']['time_patterns']):
        return 'German'
    if any(re.search(p, text, flags=re.IGNORECASE) for p in LANGUAGE_CONFIG['Chinese']['time_patterns']):
        return 'Chinese'
    return 'English'  # Fallback to English [[2]]



def detect_timezone(text, language="Hausa"):
    lang_config = LANGUAGE_CONFIG.get(language, {})
    time_patterns = lang_config.get('time_patterns', [])
    tz_patterns = lang_config.get('timezone_patterns', [])

    has_time = any(re.search(p, text, flags=re.IGNORECASE) for p in time_patterns)
    has_timezone = any(re.search(p, text, flags=re.IGNORECASE) for p in tz_patterns)

    return has_time and has_timezone  # 
def detect_calendar(text):
    cal_keywords = ['Hausa', 'German', 'Chinese', 'Gregorian']
    for k in cal_keywords:
        if k.lower() in text.lower():
            return k
    return "Gregorian"

def extract_year(date_str):
    # First try full year
    match = re.search(r"\b(1[0-9]{3}|20[0-9]{2}|21[0-9]{2})\b", date_str)
    if match:
        return int(match.group())

    # Try 2-digit year
    match_2digit = re.search(r"\b(\d{2})\b", date_str)
    if match_2digit:
        year_2digit = int(match_2digit.group())
        # Naive mapping: assume <30 = 21st century, else 20th
        return 2000 + year_2digit if year_2digit < 30 else 1900 + year_2digit

    return None


def label_period(year, date_str=None):  
    if not year:
        return "Unknown", "Unknown"
    century_num = (year - 1) // 100 + 1
    suffix = 'th' if 10 <= century_num % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(century_num % 10, 'th')

    # Build properly encoded string
    if year < 2000:
        return "Historical (Pre-2000)", f"{century_num}{suffix} Century"
    elif 2000 <= year <= 2025:
        return "Contemporary (2000–2025)", f"{century_num}{suffix} Century"
    else:
        return "Future (Post-2026)", f"{century_num}{suffix} Century"

import numpy as np
from collections import Counter

def theta(tokenized_output, baseline):
    """Compute cosine similarity between token vector and baseline."""
    t_vals = Counter(tokenized_output)
    b_vals = Counter(baseline)
    characters = list(t_vals.keys() | b_vals.keys())

    t_vect = [t_vals.get(char, 0) for char in characters]
    b_vect = [b_vals.get(char, 0) for char in characters]

    len_t = np.sqrt(sum(tv ** 2 for tv in t_vect))
    len_b = np.sqrt(sum(bv ** 2 for bv in b_vect))

    if len_t == 0 or len_b == 0:
        return 0.0

    dot = sum(tv * bv for tv, bv in zip(t_vect, b_vect))
    return dot / (len_t * len_b)

def analyze_token_semantics(tokenized_output, correct_tokens):
    token_str = " ".join(tokenized_output)
    cosine_sim = theta(tokenized_output, correct_tokens)
    R_penalty = 1 - cosine_sim

    analysis = {
        'Token Count': len(tokenized_output),
        'Splits Components': tokenized_output != correct_tokens,
        'Preserves Separators': any(sep in token_str for sep in ['-', '/', '.', ' ']),
        'Semantic Integrity': 1.0
    }

    if analysis['Splits Components']:
        analysis['Semantic Integrity'] -= 0.1  # P penalty
    if not analysis['Preserves Separators']:
        analysis['Semantic Integrity'] -= 0.1  # S penalty

    excess_tokens = max(0, len(tokenized_output) - len(correct_tokens))
    analysis['Semantic Integrity'] -= 0.05 * excess_tokens  # T penalty
    analysis['Semantic Integrity'] -= R_penalty  # R penalty (cosine deviation)

    analysis['Semantic Integrity'] = max(0.0, min(1.0, analysis['Semantic Integrity']))
    return analysis


def infer_date_format(date_str):
    format_guesses = [
        ('YYYY-MM-DD', r'^\d{4}-\d{2}-\d{2}$'),
        ('DD/MM/YYYY', r'^\d{2}/\d{2}/\d{4}$'),
        ('MM/DD/YYYY', r'^\d{2}/\d{2}/\d{4}$'),
        ('DD-MM-YYYY', r'^\d{2}-\d{2}-\d{4}$'),
	('DD.MM.YY', r'^\d{2}\.\d{2}\.\d{2}$'),
	('DD. Month YYYY', r'^\d{1,2}\. (?:Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember) \d{4}$'),
        ('Month DD, YYYY', r'^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}, \d{4}$'),
        ('DD Month YYYY', r'^\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}$'),
        ('DD.MM.YYYY', r'^\d{2}\.\d{2}\.\d{4}$'),
        ('DD. %B %Y', r'^\d{1,2}\. (?:Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember) \d{4}$'),
        ('%A, der %-d. %B %Y', r'^(?:Montag|Dienstag|Mittwoch|Donnerstag|Freitag|Samstag|Sonntag), der \d{1,2}\. (?:Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember) \d{4}$'),
	('DD.MM.YY', r'^\d{2}\.\d{2}\.\d{2}$'),
	('DD.MM.YYYY', r'^\d{2}\.\d{2}\.\d{4}$'),
	('YYYY-MM-DD', r'^\d{4}-\d{2}-\d{2}$'),
	('YYYY/MM/DD', r'^\d{4}/\d{2}/\d{2}$'),
	('YYYY.MM.DD', r'^\d{4}\.\d{2}\.\d{2}$'),
	('YYYYMMDD', r'^\d{8}$'),
	('DD. Month YYYY', r'^\d{1,2}\. (?:Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember) \d{4}$'),
        ('DD.MM.YY', r'^\d{2}\.\d{2}\.\d{2}$'),
    ]
    for fmt, pattern in format_guesses:
        if re.match(pattern, date_str):
            return fmt
    return "Unknown"



# ---------------- Analysis Loop ---------------- #
model_names = [    

	"google/mt5-large"


]

for model_name in model_names:
    print(f"Processing model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)
    results = []

    for _, row in df.iterrows():
        question = str(row.get("Question", "")).strip()
        detected_lang = detect_language(question)
        date_str = extract_date_from_question(question)
        if pd.isna(date_str): date_str = ""

        fmt = infer_date_format(date_str)
        year = extract_year(date_str)
        period, century = label_period(year, date_str=date_str)

        tokens = tokenizer.tokenize(date_str)
        
        # New cleaning logic (add only these 5 lines)
        cleaned_tokens = []
        current_number = []
        for token in tokens:
            # Fix byte-level encoding (e.g., 'Ġ' -> ' '), preserve Chinese characters
            clean_token = token.replace("Ġ", " ").strip()
            
            # Fix GPT-4's numeric splits (1904 -> ['190','4'] -> '1904')
            if clean_token.isdigit():
                current_number.append(clean_token)
            else:
                if current_number:
                    cleaned_tokens.append("".join(current_number))
                    current_number = []
                cleaned_tokens.append(clean_token)
        
        if current_number:
            cleaned_tokens.append("".join(current_number))
            
        tokens = cleaned_tokens
        # ====== END MODIFIED TOKEN PROCESSING ======
        
        correct_tokens = baseline_tokenizer(date_str, fmt)
        analysis = analyze_token_semantics(tokens, correct_tokens)
        timezone_flag = detect_timezone(question, detected_lang)

        results.append({
            "Model": model_name,
            "Date Format": fmt,
            "Language": detected_lang,
            "Year": year,
            "Time Period": period,
            "Century": century,
            "Token Count": analysis["Token Count"],
            "Tokenized Output": " ".join(tokens),
            "TokenSequence": str(tokens),
            "Semantic Integrity": analysis["Semantic Integrity"],
            "Splits Components": analysis["Splits Components"],
            "Preserves Separators": analysis["Preserves Separators"],
            "Timezone Mentioned": timezone_flag,
            "Calendar Type Detected": detect_calendar(date_str),
            "Question": question,
            "Date": date_str
        })

    # Save per model
    output_df = pd.DataFrame(results)
    safe_name = model_name.replace("/", "_")
    output_df.to_csv(f"{safe_name}_German_results.csv", index=False, encoding="utf-8-sig")
    print(f"Saved results for {model_name}!")