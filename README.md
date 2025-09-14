# 🏥 ICD Code Prediction from Medical Narratives  

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)  
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-yellow.svg)  
![Status](https://img.shields.io/badge/Status-Baseline%20Model-orange.svg)  

---

## 📖 Overview  
This project builds a **machine learning pipeline** to automatically predict **ICD (International Classification of Diseases) codes** from **unstructured medical narratives** (patient descriptions, clinical notes, etc.).  

Assigning ICD codes manually is time-consuming and error-prone. Automating this process helps improve efficiency in healthcare workflows like billing, reporting, and record-keeping.  

---

## ⚡ Features  
- 🔹 **Preprocessing with LLM** → Cleans and normalizes medical text.  
- 🔹 **TF-IDF Vectorization** → Converts text into numerical features.  
- 🔹 **Multi-label Classification** → Supports multiple ICD codes per patient.  
- 🔹 **Logistic Regression Baseline Model** → Classical ML approach for prediction.  
- 🔹 **Evaluation Metrics** → Precision, Recall, F1-score per ICD code.  
- 🔹 **Custom Predictions** → Try the model on your own narrative.  

---

## 🛠 Workflow  

### 1. Data Loading  
Dataset: `preprocessed_dataset.csv`  
- **narrative** → free-text medical case description.  
- **icd_codes** → comma-separated ICD codes (e.g., `I10,E11.9`).  

### 2. Text Preprocessing  
- Uses `preprocess_medical_text()` (LLM-powered).  
- Tasks:  
  - Normalize abbreviations.  
  - Remove irrelevant info (dispatch/transport details).  
  - Extract only symptoms & diagnoses.  

### 3. Label Preparation  
- ICD codes are split into lists:  
- Converted to a **multi-label binary matrix** using `MultiLabelBinarizer`.  

### 4. Feature Extraction  
- `TfidfVectorizer` transforms text into numeric vectors.  
- Captures word importance across the dataset.  

### 5. Train-Test Split  
- 80% training, 20% testing for fair evaluation.  

### 6. Model Training  
- **Logistic Regression** wrapped in `MultiOutputClassifier`.  
- Trains one classifier per ICD code (multi-label setup).  

### 7. Evaluation  
- Uses `classification_report` for:  
- **Precision** (accuracy of positives).  
- **Recall** (coverage of positives).  
- **F1-score** (balance between precision & recall).  

### 8. Example Prediction  
Input narrative:  
```text
Patient: 65-year-old female with persistent cough, shortness of breath, 
history of asthma and type 2 diabetes.

Cleaned Narrative: "Persistent cough, shortness of breath, asthma, type 2 diabetes."
Predicted ICD Codes: ['J45' (Asthma), 'E11.9' (Type 2 Diabetes Mellitus)]


📊 Example Output
> python icd_prediction.py
Cleaned Narrative: Persistent cough, shortness of breath, asthma, type 2 diabetes.
Predicted ICD Codes: ['J45', 'E11.9']



📂 Project Structure 
ICD_Prediction_Project/
│── icd_prediction.py        # Main pipeline script
│── preprocess.py            # LLM-based text preprocessing
│── preprocessed_dataset.csv # Example dataset
│── README.md                # Documentation


🚀 Future Improvements

⚡ Use ClinicalBERT / BioBERT embeddings instead of TF-IDF.

⚡ Replace baseline with Random Forest, XGBoost, or Deep Learning.

⚡ Add threshold tuning for probability-based predictions.

⚡ Balance rare ICD codes (merge or augment).

⚡ Expand dataset for better generalization.


## ⚠️ Limitations / Challenges  

- **Small Dataset Problem**  
  - ICD code datasets are often small and imbalanced (some codes appear very rarely).  
  - With limited samples, the model struggles to learn meaningful patterns → leading to poor precision/recall.  
  - Example: If a code appears only 2–3 times, the classifier usually predicts “not present” for it.  

- **Data Confidentiality**  
  - Real-world ICD-coded medical narratives are **highly confidential** due to patient privacy (HIPAA, GDPR, etc.).  
  - Hospitals and clinics rarely share such data publicly, making it difficult to collect larger training sets.  

- **Need for Larger Datasets**  
  - A larger, diverse dataset is crucial for improving the model’s performance.  
  - Public datasets (e.g., **MIMIC-III, MIMIC-IV**) can help, but they still require proper access agreements.  
  - Without enough data, the model remains a **baseline demo** rather than a production-ready system.  



🧑‍💻 Tech Stack
Python 🐍
Pandas, NumPy
scikit-learn
TF-IDF Vectorizer
Logistic Regression (baseline)


📜 License
This project is open-source and available under the MIT License.
