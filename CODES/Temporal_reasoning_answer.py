import pandas as pd
import subprocess
import json

def get_ollama_response(prompt, model="deepseek-r1:1.5b"):
    try:
        # Call Ollama using subprocess and pass prompt as input
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=600  # increased timeout
        )

        output = result.stdout.decode("utf-8")
        # Extract only the response (assuming plain response)
        return output.strip()
    except Exception as e:
        print(f"Error generating response: {e}")
        return ""

def generate_predictions_ollama(model_name, data_path, output_path, question_column="Question"):
    df = pd.read_excel(data_path)
    predictions = []

    for i, row in df.iterrows():
        question = str(row[question_column])

        # Add options if available
        if pd.notna(row.get('Option A')) and str(row.get('Option A')).strip():
            question += (
                f" A. {row['Option A']} "
                f"B. {row['Option B']} "
                f"C. {row['Option C']} "
                f"D. {row['Option D']}"
            )

        # Append extra fields if they exist
        extra_fields = ["Date", "fact_context", "context", "none_context", "neg_answers",
                        "Started_time", "Closed_time", "Challenges_list", "Tags_list", "Description"]

        for field in extra_fields:
            value = row.get(field)
            if pd.notna(value) and str(value).strip():
                question += f"\n[{field.replace('_', ' ').title()}]: {value}"

        if question.strip():
            answer = get_ollama_response(question, model=model_name)
        else:
            answer = ""
        predictions.append(answer)
        print(f"[{i+1}/{len(df)}] Answer: {answer[:50]}...")

    df["Prediction"] = predictions
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved predictions to: {output_path}")

# === Example Usage ===
if __name__ == "__main__":
    model_id = "deepseek-r1:1.5b"
    input_file = "englishquestionstminus.xlsx"
    output_file = "predictions_Qwenai.csv"

    generate_predictions_ollama(
        model_name=model_id,
        data_path=input_file,
        output_path=output_file
    )
