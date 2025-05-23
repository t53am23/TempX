import pandas as pd
import numpy as np
import re
from datetime import datetime
from collections import Counter
from transformers import AutoTokenizer

# Load dataset
df = pd.read_excel("Merged_raw_English_Dataset_new.xlsx") 
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
        r"\b\d{2}/\d{2}/\d{4}\b",  # DD/MM/YYYY or MM/DD/YYYY
        r"\b\d{4}-\d{2}-\d{2}\b",  # YYYY-MM-DD
        r"\b\d{2}-\d{2}-\d{4}\b",  # DD-MM-YYYY
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}, \d{4}\b",  # Month DD, YYYY
	r"\b\d{2}-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{4}\b",  # DD-Mon-YYYY
        r"\b\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\b"     # DD Month YYYY
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
def theta(tokenized_output, baseline):
    t_vals = Counter(tokenized_output)
    b_vals = Counter(baseline)
    characters = list(t_vals.keys() | b_vals.keys())
    
    # Build vectors
    t_vect = [t_vals.get(char, 0) for char in characters]
    b_vect = [b_vals.get(char, 0) for char in characters]
    
    # Calculate magnitudes
    len_t = np.sqrt(sum(tv ** 2 for tv in t_vect))
    len_b = np.sqrt(sum(bv ** 2 for bv in b_vect))
    
    # Check for zero vectors
    if len_t == 0 or len_b == 0:
        return 0.0  # 
    
    # Calculate dot product and cosine similarity
    dot = sum(tv * bv for tv, bv in zip(t_vect, b_vect))
    return dot / (len_t * len_b)

def baseline_tokenizer(date_str, format_type):
    format_map = {
        'YYYY-MM-DD': '%Y-%m-%d', 'YYYY/MM/DD': '%Y/%m/%d',
        'YYYY.MM.DD': '%Y.%m.%d', 'DD-MM-YYYY': '%d-%m-%Y',
        'DD/MM/YYYY': '%d/%m/%Y', 'MM/DD/YYYY': '%m/%d/%Y',
        'YYYYMMDD': '%Y%m%d', 'MMDDYYYY': '%m%d%Y', 'DDMMYYYY': '%d%m%Y','DD-Mon-YYYY': '%d-%b-%Y',
        'Month DD, YYYY': '%B %d, %Y', 'DD Month YYYY': '%d %B %Y',
        'Month DD YYYY': '%B %d %Y', 'YYYY/DDD': '%Y/%j',
        'DDD/YYYY': '%j/%Y', 'YYYYDDD': '%Y%j', 'DDDYYYY': '%j%Y'
    }
    if format_type not in format_map:
        return [date_str]
    try:
        date_obj = datetime.strptime(date_str, format_map[format_type])
        if format_type in ['YYYY-MM-DD', 'YYYY/MM/DD', 'YYYY.MM.DD']:
            return [date_obj.strftime('%Y'), date_str[4], date_obj.strftime('%m'), date_str[7], date_obj.strftime('%d')]
        elif format_type in ['DD-MM-YYYY', 'DD/MM/YYYY']:
            return [date_obj.strftime('%d'), date_str[2], date_obj.strftime('%m'), date_str[5], date_obj.strftime('%Y')]
        elif format_type in ['MM/DD/YYYY']:
            return [date_obj.strftime('%m'), date_str[2], date_obj.strftime('%d'), date_str[5], date_obj.strftime('%Y')]
        elif format_type == 'YYYYMMDD':
            return [date_obj.strftime('%Y'), date_obj.strftime('%m'), date_obj.strftime('%d')]
        elif format_type == 'DD-Mon-YYYY':
            return [
                date_obj.strftime('%d'), 
                '-', 
                date_obj.strftime('%b'),  # Short month (Jan/Feb)
                '-', 
                date_obj.strftime('%Y')
            ]
        else:
            return [date_str]
    except:
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

def detect_timezone(text, language="English"):
    lang_config = LANGUAGE_CONFIG.get(language, {})
    time_patterns = lang_config.get('time_patterns', [])
    tz_patterns = lang_config.get('timezone_patterns', [])

    has_time = any(re.search(p, text, flags=re.IGNORECASE) for p in time_patterns)
    has_timezone = any(re.search(p, text, flags=re.IGNORECASE) for p in tz_patterns)

    return has_time and has_timezone  # 

def detect_calendar(text):
    cal_keywords = ['Hausa', 'German', 'Chinese', 'Arabic']
    for k in cal_keywords:
        if k.lower() in text.lower():
            return k
    return "Gregorian"

def extract_year(date_str):
    match = re.search(r"\b(0?[9]\d{2}|1[0-9]\d{2}|2[0-5]\d{2})\b", date_str)
    return int(match.group()) if match else None

def label_period(year, date_str=None):  
    if not year:
        return "Unknown", "Unknown"

    # Detect Hijri dates using month names
    hijri_months = [
        'Muharram', 'Safar', 'Rabi’ul-Awwal', 'Rabi’ul-Thani',
        'Jumada al-Awwal', 'Jumada al-Thani', 'Rajab', 'Sha’ban',
        'Ramadan', 'Shawwal', 'Dhul-Qa’ada', 'Dhul-Hijja'
    ]
    is_hijri = date_str and any(month.lower() in date_str.lower() for month in hijri_months)

    # Century calculation
    century = (year - 1) // 100 + 1
    suffix = 'th' if 11 <= century % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(century % 10, 'th')

    if is_hijri:
        if year <= 1300:
            period = "Historical (1–1300 AH)"
        elif 1301 <= year <= 1446:
            period = "Contemporary (1301–1446 AH)"
        else:  # 1447+
            period = "Future (1447+ AH)"
        return period, f"{century}{suffix} Century AH"
    else:
        if year < 2000:
            return "Old Historical (Pre-2000)", f"{century}{suffix} Century"
        elif 2000 <= year <= 2025:
            return "Contemporary (2000–2025)", f"{century}{suffix} Century"
        else:
            return "Future (Post-2026)", f"{century}{suffix} Century"


def analyze_token_semantics(tokenized_output, correct_tokens):
    token_str = " ".join(tokenized_output)
    analysis = {
        'Token Count': len(tokenized_output),
        'Splits Components': tokenized_output != correct_tokens,
        'Preserves Separators': any(sep in token_str for sep in ['-', '/', '.', ' ']),
        'Semantic Integrity': 1.0
    }

    # Handle empty date case: if correct_tokens is [''], set SI to 0
    if correct_tokens == ['']:
        analysis['Semantic Integrity'] = 0.0
    else:
        if analysis['Splits Components']: 
            analysis['Semantic Integrity'] -= 0.1  # P penalty
        if not analysis['Preserves Separators']: 
            analysis['Semantic Integrity'] -= 0.1  # S penalty
        excess_tokens = max(0, len(tokenized_output) - len(correct_tokens))
        analysis['Semantic Integrity'] -= 0.05 * excess_tokens  # T penalty

    # Clamp to [0,1]
    analysis['Semantic Integrity'] = max(0.0, min(1.0, analysis['Semantic Integrity']))
    return analysis

def infer_date_format(date_str):
    format_guesses = [
        ('YYYY-MM-DD', r'^\d{4}-\d{2}-\d{2}$'),
        ('DD/MM/YYYY', r'^\d{2}/\d{2}/\d{4}$'),
        ('MM/DD/YYYY', r'^\d{2}/\d{2}/\d{4}$'),
        ('DD-MM-YYYY', r'^\d{2}-\d{2}-\d{4}$'),
        ('Month DD, YYYY', r'^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}, \d{4}$'),
	('DD-Mon-YYYY', r'^\d{2}-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{4}$'),
        ('DD Month YYYY', r'^\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}$'),
        # Add more as needed
    ]
    for fmt, pattern in format_guesses:
        if re.match(pattern, date_str):
            return fmt
    return "Unknown"


# ---------------- Analysis Loop ---------------- #
model_names = [    
	"google/mt5-small",
        "google/mt5-large"

]

for model_name in model_names:
    print(f"Processing model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)
    results = []  # Reset results per model

    for _, row in df.iterrows():
        question = str(row.get("Question", "")).strip()
        detected_lang = detect_language(question)
        date_str = extract_date_from_question(question)
        if pd.isna(date_str): date_str = ""

        fmt = infer_date_format(date_str)
        year = extract_year(date_str)
        period, century = label_period(year, date_str=date_str)

        tokens = tokenizer.tokenize(date_str)
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
    output_df.to_csv(f"{safe_name}_English_results.csv", index=False, encoding="utf-8-sig")
    print(f"Saved results for {model_name}!")