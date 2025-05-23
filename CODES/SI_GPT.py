import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Bring together tokenization results from all four languages
language_paths = {
    'English': '/content/English_tokenized_dataset_merged.xlsx',
    'Chinese': '/content/Chinese_tokenized_dataset_merged.xlsx',
    'German':  '/content/German_tokenized_dataset_merged.xlsx',
    'Hausa':   '/content/Hausa_tokenized_dataset_merged.xlsx',
}

combined_data = []

# Read each language-specific file and tag its language
for language, file_path in language_paths.items():
    data = pd.read_excel(file_path)
    data['Language'] = language
    combined_data.append(data)

# Combine everything into a single DataFrame
df = pd.concat(combined_data, ignore_index=True)

# Step 2: Double-check that all the necessary columns are included
expected_columns = ['Model', 'Date_Format', 'Language', 'Semantic_Integrity', 'Calendar_Type']
missing_columns = [col for col in expected_columns if col not in df.columns]

if missing_columns:
    raise KeyError(f"Required column(s) not found: {missing_columns}")

# Step 3: Summarize the average semantic integrity scores per language and calendar
summary_df = (
    df.groupby(['Language', 'Calendar_Type'])['Semantic_Integrity']
    .mean()
    .reset_index()
)

# Step 4: Set up a custom color palette to make the chart visually distinct
color_palette = [
    '#FF5733', '#33FF57', '#5733FF', '#FFC300',
    '#33FFFF', '#FF33FF', '#C70039'
]

# Step 5: Plot the results using a grouped bar chart
plt.figure(figsize=(16, 8))
sns.barplot(
    data=summary_df,
    x='Language',
    y='Semantic_Integrity',
    hue='Calendar_Type',
    palette=color_palette
)

plt.title('How Calendar Type Affects Semantic Integrity Across Languages (GPT-4o)', fontsize=16)
plt.xlabel('Language', fontsize=12)
plt.ylabel('Average Semantic Integrity', fontsize=12)
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()
