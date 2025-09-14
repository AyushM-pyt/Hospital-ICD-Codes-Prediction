# Step 1: Install required libraries
# pip install pandas scikit-learn nltk

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Step 2: Sample dataset (replace with your real ICD-10 dataset)
data = pd.read_csv("C:\\Users\\Ayush\\OneDrive\\Desktop\\PROJECT\\Python\\HealthCare\\extended_healthcare_dataset.csv")

df = pd.DataFrame(data)

# Step 3: Preprocessing function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalpha() and t not in stop_words]
    return " ".join(tokens)

df['clean_narrative'] = df['narrative'].apply(preprocess)

# Step 4: Multi-label binarization
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['icd_codes'])

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['clean_narrative'], y, test_size=0.2, random_state=42)

# Step 6: TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 7: Train multi-label classifier
model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
model.fit(X_train_vec, y_train)

# Step 8: Evaluate
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred, target_names=mlb.classes_))

# Step 9: Predict new narrative
def predict_icd(narrative):
    desc_clean = preprocess(narrative)
    vec = vectorizer.transform([desc_clean])
    pred = model.predict(vec)
    return mlb.inverse_transform(pred)[0]

# Example usage
new_narrative = ("MEDIC 23 DISPATCHED PRIORITY (3), NON-EMERGENT, TO METHODIST HOSPITAL  FOR TRANSPORT OF A 50 YEAR OLD FEMALE /"
"TO THE HEIGHTS AT HUEBNER. PT REQUIRED TRANSPORT FOR LONG-TERM CARE SERVICES UNAVAILABLE AT SENDING FACILITY. /"
"PT REQUIRED STRETCHER DUE TO GENERALIZED WEAKNESS. UPON ARRIVAL TO HOSPITAL NURSE REPORTS PT CAME IN FOR LIVER LABWORK/JUANDICE PT WAS FOUND SEMI-FOWLERS, A&OX(4), GCS (15), ABC'S ARE PATENT, SKIN IS WARM AND DRY WITH NORMAL PERFUSION. NO JVD, /"
"TRACHEAL DEVIATION, EDEMA, OR UNNOTED DCAP-BTLS. PT TRANSFERRED TO STRETCHER VIA DRAWSHEET AND /"
"EMTX2 THEN SECURED WITH RAILS X2 AND SEATBELTS X5. PT LOADED INTO AMBULANCE WITHOUT INCIDENT. /"
"VITAL SIGNS MONITORED THROUGHOUT TRANSPORT VIA AUTOMATED BLOOD PRESSURE CUFF AND PULSE OXIMETER AND /"
"FOUND TO BE WITHIN NORMAL LIMITS. RESPIRATIONS NORMAL. PT RELAXED IN ROUTE, EQUAL AND BILATERAL RISE AND FALL /"
"OF CHEST NOTED. UNEVENTFUL TRANSPORT. UPON ARRIVAL TO DESTINATION, PT UNLOADED FROM AMBULANCE WITHOUT INCIDENT. /"
"PT TRANSFERRED TO BED VIA DRAWSHEET AND EMTX2. GAVE REPORT TO NURSE. RECEIVING FACILITY SIGNATURE OBTAINED. MEDIC 23 DECONTAMINATED AND RETURNED TO SERVICE. EOR RUBY GARCIA EMT-B")
predicted_codes = predict_icd(new_narrative)
print("Predicted ICD-10 codes:", predicted_codes)
