import pandas as pd
import os
from ollama import Client

# Connect to local Ollama server
client = Client(host='http://localhost:11434')

def preprocess_medical_text(narrative: str) -> str:
    """
    Preprocess narrative using Ollama llama3.2:
    - Extract only symptoms and diagnoses
    - Expand abbreviations
    - Normalize noisy text
    """
    prompt = prompt = f"""
    You are a medical text assistant.
    Summarize the following narrative in 3-4 concise sentences.
    Keep only clinically important details and remove irrelevant text.
    - Expand abbreviations (Pt → Patient, ER → Emergency Room, BP → Blood Pressure, etc.) for everywhere it is used 
    - Normalize noisy or misspelled text
    - Remove dispatch or transport details

    Narrative: {narrative}
    """

    response = client.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response['message']['content']


if __name__ == "__main__":
    input_file = "medical_dataset.csv"
    output_file = "preprocessed_dataset.csv"

    # Load dataset
    df = pd.read_csv("C:\\Users\\Ayush\\OneDrive\\Desktop\\PROJECT\\Python\\HealthCare\\ICD10Codes - Candidate .csv")
    print(f"Loaded dataset with {len(df)} rows.")

    # Add patient_id if not already present
    if "patient_id" not in df.columns:
        df.insert(0, "patient_id", range(1, len(df) + 1))

    # If an output file already exists, load it to resume
    if os.path.exists(output_file):
        df_out = pd.read_csv(output_file)
        print(f"Resuming from existing output file with {len(df_out)} processed rows.")
    else:
        df_out = pd.DataFrame(columns=df.columns.tolist() + ["cleaned_text"])

    processed_ids = set(df_out["patient_id"].tolist())

    # Process only unprocessed rows
    unprocessed = df[~df["patient_id"].isin(processed_ids)]

    print(f"Processing {len(unprocessed)} new rows...")

    for _, row in unprocessed.iterrows():
        try:
            cleaned = preprocess_medical_text(row["narrative"])
        except Exception as e:
            print(f"⚠️ Error processing patient_id {row['patient_id']}: {e}")
            cleaned = "ERROR"

        row_out = row.to_dict()
        row_out["cleaned_text"] = cleaned

        # Append row to output file incrementally
        df_out = pd.concat([df_out, pd.DataFrame([row_out])], ignore_index=True)
        df_out.to_csv(output_file, index=False)  # save progress

        print(f"✅ Processed patient_id {row['patient_id']}")

    print("🎉 Preprocessing complete. Final dataset saved to preprocessed_dataset.csv")
