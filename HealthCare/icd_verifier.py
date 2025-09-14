# icd_verifier.py
from ollama import Client

# Connect to local Ollama server
client = Client()

def verify_icd_prediction(narrative: str, predicted_codes: list) -> str:
    """
    Use Ollama LLM (llama3.2) to verify ICD code predictions.
    - Input: narrative (cleaned), predicted ICD codes (list)
    - Output: verification report (confirmation, corrections, or suggestions)
    """
    prompt = f"""
    You are a medical coding verifier.
    Given the clinical narrative and a list of predicted ICD-10 codes, verify if the codes are correct.
    - Confirm if each predicted code matches the narrative.
    - Suggest corrections if some codes are inaccurate.
    - Add missing codes if necessary.
    - Output in a structured format.

    Narrative:
    {narrative}

    Predicted ICD Codes: {', '.join(predicted_codes)}
    """

    response = client.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )

    return response['message']['content']


# ------------------ Example Usage ------------------
if __name__ == "__main__":
    narrative = """
    A 65-year-old female Patient presents to the Emergency Room (ER) with symptoms of 
    persistent cough and shortness of breath. She has a history of asthma and 
    type 2 Diabetes. Her vital signs are not reported in this snippet, however, 
    her BP and other details are not mentioned.
    """
    predicted_codes = ['R68.89', 'Z74.3']  # Example output from ML model

    verification = verify_icd_prediction(narrative, predicted_codes)
    print("Verification Result:\n", verification)
