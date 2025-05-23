import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# === Step 1: Read in the data ===
# This dataset includes English questions categorized by temporal labels
data_path = "/content/English_embedding.xlsx"
df = pd.read_excel(data_path)

# Keep only rows labeled with a clear temporal category
df = df[df["Category"].isin(["Past", "Present", "Future"])].copy()

# Extract the input text and its corresponding temporal category
questions = df["Question"].tolist()
labels = df["Category"].tolist()

# Convert category names into numeric form for training the classifier
label_encoder = LabelEncoder()
numeric_labels = label_encoder.fit_transform(labels)

# === Step 2: Load the LLM and tokenizer ===
# We'll use LLaMA 3.2 (1B parameter version) to get internal representations
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Ensure padding token is defined
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load the pre-trained language model with hidden state outputs enabled
model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# === Step 3: Pull embeddings from every hidden layer ===
layer_outputs = []

for sentence in questions:
    tokenized = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        output = model(**tokenized)
    all_layers = output.hidden_states  # tuple: one tensor per layer

    # For each layer, calculate the average embedding across all tokens
    sentence_layers = [layer.mean(dim=1).squeeze().cpu().numpy() for layer in all_layers]
    layer_outputs.append(sentence_layers)

# Reshape into a NumPy array: [examples, layers, embedding_dim]
layer_outputs = np.array(layer_outputs)

# === Step 4: Train logistic regression for each layer ===
# This helps us see which layers are best at predicting past/present/future
accuracy_by_layer = []

for i in range(layer_outputs.shape[1]):
    features = layer_outputs[:, i, :]
    classifier = LogisticRegression(max_iter=500)
    classifier.fit(features, numeric_labels)
    predictions = classifier.predict(features)
    acc = accuracy_score(numeric_labels, predictions)
    accuracy_by_layer.append(acc)

# === Step 5: Visualize classification accuracy across model layers ===
plt.figure(figsize=(10, 6))
plt.plot(range(len(accuracy_by_layer)), accuracy_by_layer, marker='o')
plt.title("Accuracy of Temporal Category Prediction by Layer")
plt.xlabel("Layer Number")
plt.ylabel("Classification Accuracy")
plt.grid(True)
plt.tight_layout()
plt.show()
