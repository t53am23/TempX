import pandas as pd
import re
import random
import ast

def update_dates_in_dataset(file_path):
    try:
        data = pd.read_csv(file_path, engine='python', on_bad_lines='skip')
    except FileNotFoundError:
        print(f"Unable to locate the file: {file_path}")
        return

    # Ensure backup columns exist
    for original_col in ['original_question', 'original_date', 'original_fact_context', 'original_text_answers']:
        base_col = original_col.replace("original_", "")
        if original_col not in data.columns:
            data[original_col] = data[base_col]

    year_pattern = r"\b(19|20)\d{2}\b"

    def shift_years(text, shift_value):
        def substitute(match):
            year = int(match.group())
            new_year = year + shift_value
            return str(new_year) if 1900 <= new_year <= 2099 else str(year)
        return re.sub(year_pattern, substitute, text)

    # Define year ranges
    current_range = list(range(2000, 2026))
    future_range = list(range(2030, 2086))

    for index in data.index:
        try:
            raw_date = str(data.at[index, 'date'])
            match = re.search(year_pattern, raw_date)
            if not match:
                continue

            original_year = int(match.group())
            new_year = random.choice(current_range + future_range)
            year_diff = new_year - original_year

            for field in ['question', 'date', 'fact_context', 'context']:
                text = str(data.at[index, field])
                data.at[index, field] = shift_years(text, year_diff)

            passage = data.at[index, 'fact_context']
            matched = re.findall(r'plays for ([^\n]+?) from', passage)
            if matched:
                answer = ast.literal_eval(data.at[index, 'text_answers'])
                answer['text'] = [matched[0]]
                data.at[index, 'text_answers'] = str(answer)

            data.at[index, 'date_modified'] = True

        except Exception as e:
            print(f"Row {index} skipped due to error: {e}")
            continue

    data.to_csv('final_verified_dates.csv', index=False)
    print("✅ Date values updated successfully and saved as 'final_verified_dates.csv'.")

# Run the function
update_dates_in_dataset('/content/tempbiasQA_train.csv')
