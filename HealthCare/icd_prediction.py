# icd_prediction.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from preprocess import preprocess_medical_text  # Import our LLM preprocessor

# ------------------ Load Data ------------------
# Example dataset structure: narrative, icd_codes
# icd_codes column should have comma-separated ICD codes (e.g., "I10,E11.9")

df = pd.read_csv("C:\\Users\\Ayush\\OneDrive\\Desktop\\PROJECT\\Python\\HealthCare\\preprocessed_dataset.csv")

# Preprocess narratives with LLM
df["cleaned_narrative"] = df["narrative"].apply(preprocess_medical_text)

# Split ICD codes into list format
df["icd_codes"] = df["icd_codes"].apply(lambda x: x.split(","))

# ------------------ Feature Extraction ------------------
X = df["cleaned_narrative"]
y = df["icd_codes"]

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Convert ICD codes into multilabel format
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
y_bin = mlb.fit_transform(y)

# ------------------ Train-Test Split ------------------
X_train, X_test, y_train, y_test = train_test_split(X_vec, y_bin, test_size=0.2, random_state=42)

# ------------------ Model ------------------

model = MultiOutputClassifier(RandomForestClassifier(n_estimators=200, class_weight="balanced"))

model.fit(X_train, y_train)

# ------------------ Evaluation ------------------
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=mlb.classes_))

# ------------------ Example Prediction ------------------
sample_narrative = """
Patient: 65-year-old female with persistent cough, shortness of breath, 
history of asthma and type 2 diabetes.
"""
# Preprocess with LLM
cleaned_text = preprocess_medical_text(sample_narrative)

# Vectorize
sample_vec = vectorizer.transform([cleaned_text])

# Predict ICD codes
pred_codes = mlb.inverse_transform(model.predict(sample_vec))



print("Cleaned Narrative:", cleaned_text)
print("Predicted ICD Codes:", pred_codes)
