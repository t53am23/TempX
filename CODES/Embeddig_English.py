import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns

# === Load your dataset ===
data_path = "/content/Embedding_English_Test.xlsx"
df = pd.read_excel(data_path)

# Only keep entries labeled with valid time categories
df = df[df["Category"].isin(["Past", "Present", "Future"])]

# === Load pre-trained language model ===
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# If pad token isn't defined, fall back to EOS token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(
    model_name, output_hidden_states=True
).to(device)
model.eval()

# === Function to extract model features ===
def get_representation(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    # Mean of all hidden layers across all tokens
    hidden_layers = outputs.hidden_states
    embedding = torch.stack(hidden_layers).mean(dim=0).mean(dim=1).squeeze().cpu().numpy()

    # Mean softmax over all tokens
    prob_distribution = torch.softmax(outputs.logits, dim=-1)
    softmax_avg = prob_distribution.mean(dim=1).squeeze().cpu().numpy()

    return embedding, softmax_avg

# === Process all questions in the dataset ===
embeddings = []
softmax_outputs = []
categories = []

for _, row in df.iterrows():
    vector, prob = get_representation(row["Question"])
    embeddings.append(vector)
    softmax_outputs.append(prob)
    categories.append(row["Category"].strip().capitalize())

embeddings = np.array(embeddings)
softmax_outputs = np.array(softmax_outputs)
categories = np.array(categories)

# Organize data by time category
grouped = {}
for label in ["Past", "Present", "Future"]:
    mask = categories == label
    if np.any(mask):
        grouped[label] = (embeddings[mask], softmax_outputs[mask])
        print(f"{label}: {mask.sum()} samples")

if not grouped:
    raise ValueError("No valid samples found for the time categories.")

# Calculate group-level average embeddings and softmax vectors
mean_vectors = {k: v[0].mean(axis=0) for k, v in grouped.items()}
mean_softmax = {k: v[1].mean(axis=0) for k, v in grouped.items()}
group_labels = list(grouped.keys())

# Prepare matrices for comparison
cosine_matrix = np.zeros((len(group_labels), len(group_labels)))
kl_matrix = np.zeros_like(cosine_matrix)

for i, label_i in enumerate(group_labels):
    for j, label_j in enumerate(group_labels):
        emb_i = mean_vectors[label_i].reshape(1, -1)
        emb_j = mean_vectors[label_j].reshape(1, -1)
        cosine_matrix[i, j] = cosine_similarity(emb_i, emb_j)[0, 0]

        prob_i = mean_softmax[label_i] + 1e-10
        prob_j = mean_softmax[label_j] + 1e-10
        kl_matrix[i, j] = entropy(prob_i, prob_j)

# === Visualize results using heatmaps ===
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.heatmap(cosine_matrix, annot=True, xticklabels=group_labels, yticklabels=group_labels, cmap="Blues", ax=axes[0])
axes[0].set_title("Cosine Similarity between Categories")

sns.heatmap(kl_matrix, annot=True, xticklabels=group_labels, yticklabels=group_labels, cmap="Reds", ax=axes[1])
axes[1].set_title("KL Divergence between Categories")

plt.tight_layout()
plt.show()
