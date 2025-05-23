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

# Load the dataset from Excel
df = pd.read_excel("/content/English_embedding.xlsx")
questions = df["Question"].tolist()

# Identify the date format in each question string
def extract_date_format(text):
    regex_map = {
        "Slash (DD/MM/YYYY or MM/DD/YYYY)": r"\b\d{2}/\d{2}/\d{4}\b",
        "ISO (YYYY-MM-DD)": r"\b\d{4}-\d{2}-\d{2}\b",
        "Dash (DD-MM-YYYY)": r"\b\d{2}-\d{2}-\d{4}\b",
        "Month DD, YYYY": r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}, \d{4}\b",
        "DD-Mon-YYYY": r"\b\d{2}-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{4}\b",
        "DD Month YYYY": r"\b\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\b"
    }
    for name, pattern in regex_map.items():
        if re.search(pattern, text):
            return name
    return "Other"

df["Format"] = df["Question"].apply(extract_date_format)
formats = df["Format"].unique().tolist()

# Load LLaMA model and tokenizer
model_id = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id, output_hidden_states=True).to(
    torch.device("cuda" if torch.cuda.is_available() else "cpu"))
model.eval()

# Gather embeddings and softmax outputs
avg_embs, softmax_outputs, per_layer_embs = [], [], []
labels = df["Format"].tolist()

for question in questions:
    encoded = tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        result = model(**encoded)
    hidden = result.hidden_states
    token_embeddings = [h.mean(dim=1).squeeze().cpu().numpy() for h in hidden]
    avg_embs.append(np.stack(token_embeddings).mean(axis=0))
    per_layer_embs.append(token_embeddings)
    avg_softmax = torch.softmax(result.logits, dim=-1).mean(dim=1).squeeze().cpu().numpy()
    softmax_outputs.append(avg_softmax)

avg_embs = np.array(avg_embs)
softmax_outputs = np.array(softmax_outputs)
per_layer_embs = np.array(per_layer_embs)
label_array = np.array(labels)

# Compute average features grouped by format
grouped = {
    fmt: (
        avg_embs[label_array == fmt],
        softmax_outputs[label_array == fmt]
    )
    for fmt in formats
}
mean_embeddings = {fmt: val[0].mean(axis=0) for fmt, val in grouped.items() if len(val[0]) > 0}
mean_softmaxes = {fmt: val[1].mean(axis=0) for fmt, val in grouped.items() if len(val[1]) > 0}
valid_formats = list(mean_embeddings.keys())

# Cosine similarity and KL divergence
cos_sim_matrix = np.zeros((len(valid_formats), len(valid_formats)))
kl_matrix = np.zeros((len(valid_formats), len(valid_formats)))

for i, f1 in enumerate(valid_formats):
    for j, f2 in enumerate(valid_formats):
        cos_sim_matrix[i, j] = cosine_similarity(
            mean_embeddings[f1].reshape(1, -1),
            mean_embeddings[f2].reshape(1, -1)
        )[0][0]
        kl_matrix[i, j] = entropy(mean_softmaxes[f1] + 1e-10, mean_softmaxes[f2] + 1e-10)

# Additional temporal grouping
if "Category" not in df.columns:
    raise ValueError("Missing 'Category' column for temporal classification.")

df["Temporal_Format"] = list(zip(df["Category"], df["Format"]))
temporal_groups = {}
for (time_label, fmt), group in df.groupby("Temporal_Format"):
    idx = group.index.tolist()
    emb = avg_embs[idx]
    soft = softmax_outputs[idx]
    if len(emb) > 0:
        temporal_groups[(time_label, fmt)] = (emb, soft)

# Cosine similarity heatmaps for Past/Present/Future
for temporal in ["Past", "Present", "Future"]:
    relevant_formats = [fmt for (cat, fmt) in temporal_groups if cat == temporal]
    if len(relevant_formats) < 2:
        print(f"Not enough formats to compare in {temporal}")
        continue

    mat = np.zeros((len(relevant_formats), len(relevant_formats)))
    for i, fmt1 in enumerate(relevant_formats):
        for j, fmt2 in enumerate(relevant_formats):
            emb1 = temporal_groups[(temporal, fmt1)][0].mean(axis=0)
            emb2 = temporal_groups[(temporal, fmt2)][0].mean(axis=0)
            mat[i, j] = cosine_similarity([emb1], [emb2])[0][0]

    plt.figure(figsize=(8, 6))
    sns.heatmap(mat, xticklabels=relevant_formats, yticklabels=relevant_formats, annot=True, cmap="YlGnBu")
    plt.title(f"Cosine Similarity: {temporal} Formats")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# PCA visualization of embeddings
pca = PCA(n_components=2)
emb_2d = pca.fit_transform(avg_embs)
pca_frame = pd.DataFrame({
    "PCA1": emb_2d[:, 0],
    "PCA2": emb_2d[:, 1],
    "Format": labels
})

plt.figure(figsize=(10, 6))
sns.scatterplot(data=pca_frame, x="PCA1", y="PCA2", hue="Format", palette="Set1", s=80, edgecolor="black")
plt.title("2D PCA of Model Embeddings by Date Format")
plt.grid(True)
plt.tight_layout()
plt.show()

# Logistic regression per hidden layer
encoder = LabelEncoder()
numeric_targets = encoder.fit_transform(labels)

layer_accuracy = []
for layer_index in range(per_layer_embs.shape[1]):
    features = per_layer_embs[:, layer_index, :]
    model = LogisticReg
