
'''This Model is overfitted due to less data samples..'''

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from icd10 import ICD10

# Load sample dataset
data = pd.read_csv("C:\\Users\\Ayush\\OneDrive\\Desktop\\PROJECT\\Python\\HealthCare\\ICD10Codes - Candidate .csv")

# Define function to preprocess narrative data
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    return ' '.join(tokens)

# Preprocess text data
data['narrative'] = data['narrative'].apply(preprocess_text)

# Define TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit TF-IDF vectorizer to narrative data
X = vectorizer.fit_transform(data['narrative'])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, data['icd_codes'], test_size=0.2, random_state=42)

# Train RF classifier on training data
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)

# Evaluate 
y_pred = rfc.predict(X_test)

print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# Define function to predict ICD code from narrative
def predict_icd(narrative):
    processed_text = preprocess_text(narrative)
    vector = vectorizer.transform([processed_text])
    prediction = rfc.predict(vector)[0]
    return prediction

# Test the model on a sample narrative
'''narrative = ("Medic 23 dispatched to north east Methodist for a 68 yo female cc. Requires stretcher because of /"
"trach collar oxygen and weakness.Being transferred for long term care not available at sending facility./"
"Medic 23 arrived on scene and spoke with nurse. Nurse reports put pt back on vent because she de stated during /"
"therapy but is stablized and being put back on the trach collar. Came in for cellulitis of left leg /"
"dischargeing on daily lasix left upper midline going for a ten day course of antibiotics. Patient non verbal /"
"but alert and oriented. Collar setting 15 lpm at 50%. medic 23 made patient contact patient found lying in /"
"hospital bed laying semi fowlers aox4 gcs 15 no complaints of pain. Pt transferred from hospital bed to /"
"stretcher via 2 person slide sheet without incident. Pt secured to stretcher via 2 shoulder straps and 3 /"
"straps x 2 side rails. Patient signed for consent for treatment and transport. Pt transported and loaded into /"
"ambulance via stretcher without incident. Pt vitals monitored throughout transport without incident. Medic 23 /"
"arrived at destination unloaded pt and transported to patient room via stretcher without incident. Pt transferred from stretcher to pt bed via 2 person draw sheet without incident report and /"
"paperwork given to nurse. Signature obtained . Medic 23 cleared and returned to service eor Roger Craig EMT-B")'''

narrative = "A 55-year-old male presents with persistent chest pain radiating to the left arm, shortness of breath on exertion, and occasional dizziness. He reports a history of hypertension and high cholesterol. On examination, blood pressure is elevated at 150/95 mmHg, heart rate is 88 bpm, and ECG shows ST-segment depression. Laboratory tests reveal mildly elevated troponin levels."
icd_code = predict_icd(narrative)
print(f"Predicted ICD code: {icd_code}")


