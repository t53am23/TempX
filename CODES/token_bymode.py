import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Define file paths for each language-specific dataset
data_sources = {
    'English': '/content/English_tokenized_dataset_merged.xlsx',
    'Chinese': '/content/Chinese_tokenized_dataset_merged.xlsx',
    'German':  '/content/German_tokenized_dataset_merged.xlsx',
    'Hausa':   '/content/Hausa_tokenized_dataset_merged.xlsx',
}

# Step 2: Read and tag each dataset with its corresponding language
combined_data = []
for language, filepath in data_sources.items():
    temp_df = pd.read_excel(filepath)
    temp_df['Language'] = language
    combined_data.append(temp_df)

# Step 3: Merge all datasets into one master DataFrame
df_all = pd.concat(combined_data, ignore_index=True)

# Step 4: Compute per-model, per-language averages for key metrics
summary_stats = (
    df_all
    .groupby(['Model', 'Language'])[['Semantic_Integrity', 'Token_Count']]
    .mean()
    .reset_index()
)

# Step 5: Pivot tables to get a format suitable for horizontal bar charts
si_table = summary_stats.pivot(index='Model', columns='Language', values='Semantic_Integrity')
tc_table = summary_stats.pivot(index='Model', columns='Language', values='Token_Count')

# Step 6: Create visualizations to compare model behavior across languages
fig, (ax_si, ax_tc) = plt.subplots(2, 1, figsize=(10, 14))

# Plot 1: Semantic Integrity scores by model and language
si_table.plot(kind='barh', ax=ax_si, colormap='Set2', width=0.8)
ax_si.set_title('Semantic Integrity by Model Across Languages', fontsize=14)
ax_si.set_xlabel('Average Semantic Integrity')
ax_si.legend(title='Language', bbox_to_anchor=(1, 1))

# Plot 2: Token Count by model and language
tc_table.plot(kind='barh', ax=ax_tc, colormap='Set3', width=0.8)
ax_tc.set_title('Token Count by Model Across Languages', fontsize=14)
ax_tc.set_xlabel('Average Token Count')
ax_tc.legend(title='Language', bbox_to_anchor=(1, 1))

# Final layout adjustment
plt.tight_layout()
plt.show()
