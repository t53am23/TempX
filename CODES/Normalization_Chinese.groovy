import pandas as pd
import random
import re
from datetime import datetime

# Chinese digit symbols mapped to Arabic numerals
digit_to_chinese = {
    '0': '〇', '1': '一', '2': '二', '3': '三', '4': '四',
    '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'
}

# Convert integer 1–10 to its Chinese form
def to_chinese_digit(n):
    digits = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]
    return digits[n - 1] if 1 <= n <= 10 else ""

# Render full year as a Chinese numeral string
def format_year_chinese(year):
    return ''.join(digit_to_chinese[char] for char in str(year))

# Create a random Chinese lunar date
def random_lunar_date(year=None):
    months = ["正月", "二月", "三月", "四月", "五月", "六月",
              "七月", "八月", "九月", "十月", "冬月", "腊月"]

    early_days = [f"初{to_chinese_digit(i)}" for i in range(1, 11)]
    mid_days = ["十一", "十二", "十三", "十四", "十五", "十六", "十七", "十八", "十九", "二十"]
    late_days = ["廿一", "廿二", "廿三", "廿四", "廿五", "廿六", "廿七", "廿八", "廿九", "三十"]

    day = random.choice(early_days + mid_days + late_days)
    month = random.choice(months)
    chosen_year = year if year else random.choice([1940, 2024, 1430, 2005, 2010, 2030, 1890, 2025, 2080, 2050, 2040, 2030, 1740, 1720, 2000, 1190])
    return f"农历{format_year_chinese(chosen_year)}{month}{day}"

# Replace matching date strings with lunar-style equivalents
def replace_dates(text, chance=0.3):
    patterns = [
        r"\d{4}-\d{2}-\d{2}",
        r"\d{4}年\d{1,2}月\d{1,2}日",
        r"\d{2}/\d{2}/\d{4}",
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}, \d{4}",
        r"农历\d{4}年\d{1,2}月\d{1,2}日"
    ]

    found = re.findall('|'.join(patterns), text)
    for match in found:
        if random.random() < chance:
            new_date = random_lunar_date()
            text = text.replace(match, new_date)
    return text

# Apply to full dataset
def apply_lunar_transformation(input_file, output_file):
    data = pd.read_excel(input_file)
    data['Question'] = data['Question'].apply(lambda q: replace_dates(str(q)))
    data.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Dataset updated and saved to {output_file}")

# Uncomment and provide file paths to use:
# apply_lunar_transformation("path_to_input.xlsx", "path_to_output.csv")
