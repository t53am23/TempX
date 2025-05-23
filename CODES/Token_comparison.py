import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # Optional but helpful for styling

# Step 1: Define the file paths for datasets in different languages
language_paths = {
    'English': '/content/English_tokenized_dataset_merged.xlsx',
    'Chinese': '/content/Chinese_tokenized_dataset_merged.xlsx',
    'German':  '/content/German_tokenized_dataset_merged.xlsx',
    'Hausa':   '/content/Hausa_tokenized_dataset_merged.xlsx',
}

# Step 2: Read and label each file, then combine them into a single DataFrame
language_frames = []
for language, file in language_paths.items():
    df = pd.read_excel(file)
    df['Language'] = language  # Add a new column to indicate language source
    language_frames.append(df)

# Combine all datasets into one table for analysis
full_data = pd.concat(language_frames, ignore_index=True)

# Step 3: Compute average token count for each model
average_token_counts = full_data.groupby('Model')['Token_Count'].mean()
model_names = average_token_counts.index.tolist()

# Step 4: Create a horizontal bar chart to compare token counts
plt.figure(figsize=(14, 8))
plt.barh(model_names, average_token_counts.values, align='center', color='skyblue')

# Add axis labels and chart title
plt.xlabel('Average Token Count')
plt.ylabel('Model')
plt.title('Token Count Comparison Across Models and Languages')

# Clean up layout and render the plot
plt.tight_layout()
plt.show()
