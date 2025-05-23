import pandas as pd
from googletrans import Translator
from tqdm import tqdm
from itertools import product

# Load datasets
tot_df = pd.read_excel('/content/TRAM dataset.xlsx')
variations_df = pd.read_excel("/content/date_time_variations.xlsx")
timezone_df = pd.read_excel("/content/timezone_info.xlsx")

# Clean column names by removing leading/trailing spaces
tot_df.columns = tot_df.columns.str.strip()
variations_df.columns = variations_df.columns.str.strip()
timezone_df.columns = timezone_df.columns.str.strip()



os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/content/refreshing-cat-454219-e7-22c66e989629.json"

# Initialize translator
translator = Translator()

# Language mapping
lang_map = {"zh-cn": "Chinese", "ar"}

# Cache for translations to avoid redundant API calls
translation_cache = {}

def apply_date_format(text, date_formats):
    """Apply dynamic date formatting based on the target language."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    for date_type, date_str in date_formats.items():
        text = text.replace(f"{{{{{date_type}}}}}", str(date_str))
    return text

def translate_batch(texts, lang, date_formats, timezone_info):
    """
    Translate a batch of texts and return the translated texts.
    Uses caching to avoid redundant translations.
    """
    # Apply date formatting to all texts
    formatted_texts = [apply_date_format(text, date_formats) for text in texts]

    # Check cache for each formatted text
    cached_results = []
    uncached_indices = []
    uncached_texts = []

    for i, text in enumerate(formatted_texts):
        cache_key = (text, lang, tuple(date_formats.items()))
        if cache_key in translation_cache:
            cached_results.append(translation_cache[cache_key])
        else:
            cached_results.append(None)
            uncached_indices.append(i)
            uncached_texts.append(text)

    # Translate uncached texts in a single batch
    if uncached_texts:
        translated_texts = translator.translate(uncached_texts, dest=lang)
        for i, t in zip(uncached_indices, translated_texts):
            result = f"{t.text} ({timezone_info['Primary Time Zone(s)']}, {timezone_info['UTC Offset']})"
            translation_cache[(formatted_texts[i], lang, tuple(date_formats.items()))] = result
            cached_results[i] = result

    return cached_results

def translate_large_batch(texts, lang, date_formats, timezone_info):
    """Translate large batches by splitting into chunks of 100."""
    chunk_size = 100
    translated_texts = []
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i + chunk_size]
        translated_chunk = translate_batch(chunk, lang, date_formats, timezone_info)
        translated_texts.extend(translated_chunk)
    return translated_texts

# Precompute all unique texts for translation
unique_texts = set(tot_df["Question"]).union(set(tot_df["Answer"]))
for column in ["Option A", "Option B", "Option C", "Option D"]:
    unique_texts.update(tot_df[column])

# Process and save results
results = []

# Wrap the outer loop for language mapping with tqdm
for lang_code, lang_name in tqdm(lang_map.items(), desc="Languages", unit="language"):
    lang_variations = variations_df[variations_df["Language"] == lang_name]
    lang_timezone = timezone_df[timezone_df["Language"] == lang_name].iloc[0]

    # Precompute translations for all unique texts
    date_formats = {}
    for _, date_row in lang_variations.iterrows():
        date_formats.update({date_row["Calendar System"]: date_row["Date Example (Native Script)"]})

    # Translate all unique texts once
    translated_texts = translate_large_batch(list(unique_texts), lang_code, date_formats, lang_timezone)

    # Create a mapping of original texts to translated texts
    text_mapping = dict(zip(unique_texts, translated_texts))

    # Precompute all combinations to reduce nested loops
    all_combinations = list(product(tot_df.iterrows(), lang_variations.iterrows()))
    total_iterations = len(all_combinations)

    # Process all combinations
    for (_, tot_row), (_, date_row) in tqdm(all_combinations, desc="Date Variations", unit="variation", leave=False):
        date_formats = {date_row["Calendar System"]: date_row["Date Example (Native Script)"]}

        # Use precomputed translations
        translated_question = text_mapping[tot_row["Question"]]
        translated_answer = text_mapping[tot_row["Answer"]]
        translated_options = {col: text_mapping[tot_row[col]] for col in ["Option A", "Option B", "Option C", "Option D"]}

        # Collect results
        translated_row = {
            "Original_Question": tot_row["Question"],
            "Original_Answer": tot_row["Answer"],
            "Formatted_Original_Answer": apply_date_format(tot_row["Answer"], date_formats),
            "Language": lang_name,
            "TimeZone": lang_timezone["Primary Time Zone(s)"],
            "UTCOffset": lang_timezone["UTC Offset"],
            "Translated_Question": translated_question,
            "Translated_Answer": translated_answer,
        }
        for column in ["Option A", "Option B", "Option C", "Option D"]:
            translated_row[f"Translated_{column}"] = translated_options[column]
        results.append(translated_row)

# Save the final results with validation columns
final_columns = [
    "Original_Question",
    "Original_Answer",
    "Formatted_Original_Answer",
    "Language",
    "TimeZone",
    "UTCOffset",
    "Translated_Question",
    "Translated_Answer",
    *[f"Translated_{col}" for col in ["Option A", "Option B", "Option C", "Option D"]]
]
df_results = pd.DataFrame(results)[final_columns]
df_results.to_csv("translated_TRAM_questions.csv", index=False, encoding='utf-8')

print("File saved: translated_TRAM_questions.csv")

from google.colab import files
files.download("translated_TRAM_questions.csv")