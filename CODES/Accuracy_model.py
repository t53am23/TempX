import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Accuracy data for questions about time across four languages
# Each includes results for past, present, and future questions

# Results for German
german = {
    'Time': ['Past', 'Present', 'Future'],
    'Deepseek 5b': [60.00, 40.00, 0.00],
    'Ollama3 1b': [86.47, 12.03, 1.50]
}
df_german = pd.DataFrame(german).set_index('Time')

# Results for Hausa
hausa = {
    'Time': ['Past', 'Present', 'Future'],
    'Deepseek 5b': [84.21, 10.53, 5.26],
    'Ollama3 1b': [100.00, 0.00, 0.00]
}
df_hausa = pd.DataFrame(hausa).set_index('Time')

# Results for English
english = {
    'Time': ['Past', 'Present', 'Future'],
    'Deepseek 5b': [90.57, 9.43, 0.00],
    'Ollama3 1b': [80.00, 20.00, 0.00]
}
df_english = pd.DataFrame(english).set_index('Time')

# Results for Chinese
chinese = {
    'Time': ['Past', 'Present', 'Future'],
    'Deepseek 5b': [87.50, 12.50, 0.00],
    'Ollama3 1b': [48.39, 45.16, 6.45]
}
df_chinese = pd.DataFrame(chinese).set_index('Time')

# Colors for each time category
time_colors = {
    'Past': '#2ca02c',    # green
    'Present': '#d62728', # red
    'Future': '#ff7f0e'   # orange
}

# This function draws a chart for each language
def draw_chart(df, language):
    fig, ax = plt.subplots(figsize=(10, 6))
    models = df.columns.tolist()
    time_parts = df.index.tolist()
    colors = [time_colors[t] for t in time_parts]
    labels = models
    y_pos = np.arange(len(models))
    bar_height = 0.4

    for i, model in enumerate(models):
        left = 0
        for j, time in enumerate(time_parts):
            percent = df.loc[time, model]
            if percent > 0:
                ax.barh(y_pos[i], percent, height=bar_height, left=left,
                        color=colors[j], label=time if i == 0 else "")
                ax.text(left + percent / 2, y_pos[i], f"{percent:.1f}%", va='center', ha='center', fontsize=9)
                left += percent

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Percentage of Correct Answers")
    ax.set_xlim(0, 100)
    ax.set_title(f"Correct Answers by Time Type in {language}", fontsize=14, weight='bold')

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[k]) for k in range(len(time_parts))]
    ax.legend(handles, time_parts, loc='center left', bbox_to_anchor=(1, 0.5), title="Time Type")

    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.show()

# Draw charts for each language
draw_chart(df_german, "German")
draw_chart(df_hausa, "Hausa")
draw_chart(df_english, "English")
draw_chart(df_chinese, "Chinese")
