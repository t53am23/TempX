import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Let's look at how two models performed when answering time-related questions
# We break this down by past, present, and future events in four different languages

# Accuracy rates (as percentages) for each time category and language

# German
german = {
    'Period': ['Past', 'Present', 'Future'],
    'Deepseek 5b': [60.0, 40.0, 0.0],
    'Ollama3 1b': [86.47, 12.03, 1.5]
}
df_german = pd.DataFrame(german).set_index('Period')

# Hausa
hausa = {
    'Period': ['Past', 'Present', 'Future'],
    'Deepseek 5b': [84.21, 10.53, 5.26],
    'Ollama3 1b': [100.0, 0.0, 0.0]
}
df_hausa = pd.DataFrame(hausa).set_index('Period')

# English
english = {
    'Period': ['Past', 'Present', 'Future'],
    'Deepseek 5b': [90.57, 9.43, 0.0],
    'Ollama3 1b': [80.0, 20.0, 0.0]
}
df_english = pd.DataFrame(english).set_index('Period')

# Chinese
chinese = {
    'Period': ['Past', 'Present', 'Future'],
    'Deepseek 5b': [87.5, 12.5, 0.0],
    'Ollama3 1b': [48.39, 45.16, 6.45]
}
df_chinese = pd.DataFrame(chinese).set_index('Period')

# Let's combine everything into one table
all_data = []

# A simple helper to collect and label data by model and language
def add_data(df, language):
    for model in df.columns:
        for period in df.index:
            all_data.append({
                'Model-Lang': f"{model} ({language})",
                'Period': period,
                'Score': df.loc[period, model]
            })

# Add each language one by one
add_data(df_english, 'English')
add_data(df_german, 'German')
add_data(df_hausa, 'Hausa')
add_data(df_chinese, 'Chinese')

# Make a new table that shows past, present, and future scores side by side
df_all = pd.DataFrame(all_data)
pivot_df = df_all.pivot(index='Model-Lang', columns='Period', values='Score').fillna(0)
pivot_df = pivot_df[['Past', 'Present', 'Future']]  # Keep things in logical order

# Choose colors to represent each time period
colors = {'Past': '#2ca02c', 'Present': '#d62728', 'Future': '#ff7f0e'}

# Now we draw the chart
fig, ax = plt.subplots(figsize=(12, 8))
positions = np.arange(len(pivot_df))
height = 0.8
left_pos = np.zeros(len(pivot_df))

# Build the bars
for period in ['Past', 'Present', 'Future']:
    scores = pivot_df[period].values
    ax.barh(positions, scores, left=left_pos, height=height, color=colors[period], label=period)
    for i, score in enumerate(scores):
        if score > 0:
            ax.text(left_pos[i] + score / 2, positions[i], f"{score:.1f}%", ha='center', va='center',
                    color='white' if score < 10 else 'black', fontsize=8)
    left_pos += scores

# Finish up the plot
ax.set_yticks(positions)
ax.set_yticklabels(pivot_df.index)
ax.invert_yaxis()
ax.set_xlabel("Correct Answer Rate (%)")
ax.set_xlim(0, 100)
ax.set_title("Correct Answer Breakdown by Time Period and Language", fontsize=14, weight='bold')

# Add a legend
legend_items = [plt.Rectangle((0, 0), 1, 1, color=colors[p]) for p in colors]
ax.legend(legend_items, list(colors.keys()), title="Time Period", loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout(rect=[0, 0, 0.8, 1])
plt.show()
