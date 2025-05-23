import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cross-lingual summary data and clean the column headers
df = pd.read_csv("/content/Cross lingual__Model_summary.csv")
df.columns = df.columns.str.strip()
df.dropna(subset=["Model", "TC", "SI"], inplace=True)

# Step 1: Group data by model to compute bias-related statistics
summary = df.groupby("Model").agg({
    "SI": ["mean", "std", lambda s: s.max() - s.min()],
    "TC": ["mean", "std", lambda s: s.max() - s.min()],
    "Language": lambda s: s.nunique()
}).reset_index()

# Rename multi-level column headers for clarity
summary.columns = [
    "Model", "SI_Average", "SI_StdDev", "SI_Range",
    "TC_Average", "TC_StdDev", "TC_Range", "Num_Languages"
]

# Step 2: Identify best and worst performing languages per model (based on SI)
best_per_lang = df.loc[df.groupby("Model")["SI"].idxmax()][["Model", "Language"]]
best_per_lang.columns = ["Model", "Best_Language"]

worst_per_lang = df.loc[df.groupby("Model")["SI"].idxmin()][["Model", "Language"]]
worst_per_lang.columns = ["Model", "Worst_Language"]

# Merge those with the summary dataframe
summary = summary.merge(best_per_lang, on="Model").merge(worst_per_lang, on="Model")

# Step 3: Compute a weighted score to quantify model bias
summary["Bias_Score"] = (
    0.7 * summary["SI_Range"] +
    0.3 * summary["SI_StdDev"] +
    0.5 * summary["TC_Range"]
)

# Step 4: Round numeric values to 3 decimal places for readability
numeric_fields = [
    "SI_Average", "SI_StdDev", "SI_Range",
    "TC_Average", "TC_StdDev", "TC_Range", "Bias_Score"
]
summary[numeric_fields] = summary[numeric_fields].round(3)

# Step 5: Sort models by their calculated bias score (descending order)
summary = summary.sort_values("Bias_Score", ascending=False)

# Save final summary to file
summary.to_csv("model_language_bias_analysis.csv", index=False, encoding="utf-8-sig")
print("✅ Analysis complete. Results saved to 'model_language_bias_analysis.csv'.")

# ───── VISUALIZATION ───── #

# Plot 1: Bar chart showing the 10 most biased models
plt.figure(figsize=(10, 6))
sns.barplot(data=summary.head(10), x="Bias_Score", y="Model", palette="viridis")
plt.title("Top 10 Most Biased Models by Bias Score", fontsize=14)
plt.xlabel("Bias Score")
plt.ylabel("Model")
plt.tight_layout()
plt.show()

# Plot 2: Scatter showing SI vs TC with color based on bias and size based on language count
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=summary,
    x="SI_Average",
    y="TC_Average",
    hue="Bias_Score",
    size="Num_Languages",
    sizes=(50, 200),
    palette="coolwarm",
    legend=False
)
plt.title("Semantic Integrity vs Translation Clarity", fontsize=14)
plt.xlabel("Mean SI")
plt.ylabel("Mean TC")
plt.tight_layout()
plt.show()

# Plot 3: Heatmap showing how the bias-related metrics correlate with one another
plt.figure(figsize=(10, 8))
metrics_only = summary[numeric_fields]
sns.heatmap(metrics_only.corr(), annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlation Between Semantic and Translation Metrics", fontsize=14)
plt.tight_layout()
plt.show()
