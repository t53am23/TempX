# Load all required libraries
import torch
import pandas as pd
import numpy as np
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Read the Excel file containing questions
df = pd.read_excel("/content/English_embedding.xlsx")
texts = df["Question"].tolist()

# Identify the type of date format in each question
def detect_format(text):
    patterns = {
        "Slash (DD/MM/YYYY or MM/DD/YYYY)": r"\b\d{2}/\d{2}/\d{4}\b",
        "ISO (YYYY-MM-DD)": r"\b\d{4}-\d{2}-\d{2}\b",
        "Dash (DD-MM-YYYY)": r"\b\d{2}-\d{2}-\d{4}\b",
        "Month DD, YYYY": r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}, \d{4}\b",
        "DD-Mon-YYYY": r"\b\d{2}-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{4}\b",
        "DD Month YYYY": r"\b\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\b"
    }
    for name, pat in patterns.items():
        if re.search(pat, text):
            return name
    return "Other"

df["Format"] = df["Question"].apply(detect_format)
df = df[df["Format"] != "Other"].reset_index(drop=True)
texts = df["Question"].tolist()
formats = df["Format"].unique().tolist()

# Load the tokenizer and model
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    output_hidden_states=True
).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
model.eval()

# Extract features from the model
layer_outputs = []
average_vectors = []
softmax_vectors = []
labels = df["Format"].tolist()

for sentence in texts:
    encoded = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        result = model(**encoded)

    layers = result.hidden_states
    # Average token representation for each layer
    per_layer = [h.mean(dim=1).squeeze().cpu().numpy() for h in layers]
    layer_outputs.append(per_layer)
    average_vectors.append(np.stack(per_layer).mean(axis=0))
    avg_softmax = torch.softmax(result.logits, dim=-1).mean(dim=1).squeeze().cpu().numpy()
    softmax_vectors.append(avg_softmax)

layer_outputs = np.array(layer_outputs)
average_vectors = np.array(average_vectors)
softmax_vectors = np.array(softmax_vectors)
label_array = np.array(labels)

# Organize results by format
grouped_data = {f: (average_vectors[label_array == f], softmax_vectors[label_array == f]) for f in formats}
group_avg_emb = {f: v[0].mean(axis=0) for f, v in grouped_data.items()}
group_avg_soft = {f: v[1].mean(axis=0) for f, v in grouped_data.items()}

# Compute cosine and KL divergence between formats
cosine_matrix = np.zeros((len(formats), len(formats)))
kl_matrix = np.zeros((len(formats), len(formats)))

for i, fi in enumerate(formats):
    for j, fj in enumerate(formats):
        cosine_matrix[i, j] = cosine_similarity([group_avg_emb[fi]], [group_avg_emb[fj]])[0][0]
        kl_matrix[i, j] = entropy(group_avg_soft[fi] + 1e-10, group_avg_soft[fj] + 1e-10)

# If available, compare formats within temporal categories
if "Category" in df.columns:
    df["Group"] = list(zip(df["Category"], df["Format"]))
    temp_groups = {}
    for (cat, fmt), group in df.groupby("Group"):
        idx = group.index.tolist()
        temp_groups[(cat, fmt)] = (average_vectors[idx], softmax_vectors[idx])

    temp_avg_emb = {k: v[0].mean(axis=0) for k, v in temp_groups.items()}
    for temporal in ["Past", "Present", "Future"]:
        selected = [f for (c, f) in temp_groups if c == temporal]
        if len(selected) < 2:
            continue
        mat = np.zeros((len(selected), len(selected)))
        for i, f1 in enumerate(selected):
            for j, f2 in enumerate(selected):
                mat[i, j] = cosine_similarity(
                    [temp_avg_emb[(temporal, f1)]],
                    [temp_avg_emb[(temporal, f2)]]
                )[0][0]
        plt.figure(figsize=(8, 6))
        sns.heatmap(mat, annot=True, xticklabels=selected, yticklabels=selected, cmap="YlGnBu")
        plt.title(f"{temporal} Formats")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# PCA visualization
pca = PCA(n_components=2)
pca_results = pca.fit_transform(average_vectors)
plot_df = pd.DataFrame({
    "X": pca_results[:, 0],
    "Y": pca_results[:, 1],
    "Format": labels
})
plt.figure(figsize=(10, 6))
sns.scatterplot(data=plot_df, x="X", y="Y", hue="Format", palette="Set1", s=80)
plt.title("Embedding PCA by Format")
plt.grid(True)
plt.tight_layout()
plt.show()

# Evaluate how well each hidden layer can identify the format
encoder = LabelEncoder()
numeric_labels = encoder.fit_transform(labels)
layer_accuracy = []

for i in range(layer_outputs.shape[1]):
    X = layer_outputs[:, i, :]
    y = numeric_labels
    clf = LogisticRegression(max_iter=500)
    clf.fit(X, y)
    preds = clf.predict(X)
    acc = accuracy_score(y, preds)
    layer_accuracy.append(acc)

plt.figure(figsize=(10, 6))
plt.plot(range(len(layer_accuracy)), layer_accuracy, marker='o')
plt.title("Accuracy by Layer (Format Prediction)")
plt.xlabel("Layer Index")
plt.ylabel("Accuracy")
plt.grid(True)
plt.tight_layout()
plt.show()

# Optional: compare embedding similarities for temporal labels
temporal_tags = ["Past", "Present", "Future"]
temporal_vectors = {tag: grouped_data[tag][0].mean(axis=0) for tag in temporal_tags if tag in grouped_data}
matrix = np.zeros((len(temporal_vectors), len(temporal_vectors)))
for i, ti in enumerate(temporal_vectors):
    for j, tj in enumerate(temporal_vectors):
        matrix[i, j] = cosine_similarity([temporal_vectors[ti]], [temporal_vectors[tj]])[0][0]

plt.figure(figsize=(6, 5))
sns.heatmap(matrix, annot=True, xticklabels=temporal_vectors.keys(), yticklabels=temporal_vectors.keys(), cmap="PuBu")
plt.title("Temporal Class Comparison")
plt.tight_layout()
plt.show()
