import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Set up the results for each language and question type

# Chinese model results
chinese = {
    'Type of Question': ['Temporal Arithmetic', 'Calendar difference', 'Time zone Conversion', 'Multi-hop Reasoning'],
    'Deepseek 5b': [87.50, 12.50, 0.00, 0.00],
    'Ollama3 1b': [58.06, 6.45, 9.68, 25.81]
}
df_chinese = pd.DataFrame(chinese).set_index('Type of Question')

# English model results
english = {
    'Type of Question': ['Temporal Arithmetic', 'Calendar and Time Conversion', 'Multi-hop Reasoning'],
    'Deepseek 5b': [69.81, 24.53, 5.66],
    'Ollama3 1b': [20.00, 63.33, 16.67]
}
df_english = pd.DataFrame(english).set_index('Type of Question')

# Hausa model results
hausa = {
    'Type of Question': ['Temporal Arithmetic', 'Calendar Conversion', 'Time Zone Conversion', 'Multi-hop Reasoning'],
    'Deepseek 5b': [73.68, 15.79, 5.26, 5.26],
    'Ollama3 1b': [75.00, 12.50, 12.50, 0.00]
}
df_hausa = pd.DataFrame(hausa).set_index('Type of Question')

# Step 2: Organize everything into one dataset for plotting

all_data = []

# Function to prepare the data for each language and model
def collect_data(df, lang):
    for model in df.columns:
        for q_type in df.index:
            all_data.append({
                'Label': f"{model} ({lang})",
                'Question Type': q_type,
                'Score': df.loc[q_type, model]
            })

# Collect data for each language
collect_data(df_english, 'English')
collect_data(df_hausa, 'Hausa')
collect_data(df_chinese, 'Chinese')

df_all = pd.DataFrame(all_data)

# Reorganize it so we can easily make a grouped bar chart
df_chart = df_all.pivot_table(index='Label', columns='Question Type', values='Score', fill_value=0)

# Make sure we only include question types that exist in the data
question_order = [
    'Temporal Arithmetic', 'Calendar difference', 'Calendar and Time Conversion',
    'Calendar Conversion', 'Time zone Conversion', 'Time Zone Conversion', 'Multi-hop Reasoning'
]
question_order = [q for q in question_order if q in df_chart.columns]
df_chart = df_chart[question_order]

# Step 3: Choose some clear colors for each type of question
colors = {
    'Temporal Arithmetic': '#2ca02c',        # Green
    'Calendar difference': '#d62728',        # Red
    'Calendar and Time Conversion': '#d62728',
    'Calendar Conversion': '#d62728',
    'Time zone Conversion': '#ff7f0e',       # Orange
    'Time Zone Conversion': '#ff7f0e',
    'Multi-hop Reasoning': '#17becf'         # Cyan
}

# Step 4: Draw the chart
fig, ax = plt.subplots(figsize=(12, 8))
positions = np.arange(len(df_chart))
bar_height = 0.8
left = np.zeros(len(df_chart))

for q_type in question_order:
    scores = df_chart[q_type].values
    ax.barh(positions, scores, height=bar_height, left=left, color=colors[q_type], label=q_type)

    for i, (score, start) in enumerate(zip(scores, left)):
        if score > 0:
            ax.text(start + score / 2, positions[i], f"{score:.1f}%", ha='center', va='center',
                    color='white' if score < 10 else 'black', fontsize=8)
    left += scores

# Final touches
ax.set_yticks(positions)
ax.set_yticklabels(df_chart.index)
ax.invert_yaxis()
ax.set_xlabel("Correct Answer Rate (%)")
ax.set_xlim(0, 100)
ax.set_title("Model Accuracy by Question Type and Language", fontsize=14, weight='bold')

# Create the legend
legend_items = [plt.Rectangle((0, 0), 1, 1, color=colors[q]) for q in question_order]
ax.legend(legend_items, question_order, title="Question Type", loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout(rect=[0, 0, 0.8, 1])
plt.show()
