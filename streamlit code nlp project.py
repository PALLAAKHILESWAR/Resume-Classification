import streamlit as st
import pandas as pd
import os
import zipfile
import tempfile
import pickle
from docx import Document
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

st.title("Resume Classifier (.docx only)")

MODEL_PATH = "nb_model.pkl"
VEC_PATH = "vectorizer.pkl"

# Local path to resumes folder (FIXED path)
RESUME_DIR = r"C:\Users\mouni\Downloads\Resumes_Docx"

# Read resumes from local directory
data = []
for folder in os.listdir(RESUME_DIR):
    folder_path = os.path.join(RESUME_DIR, folder)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith(".docx"):
                file_path = os.path.join(folder_path, file)
                try:
                    doc = Document(file_path)
                    content = "\n".join([p.text for p in doc.paragraphs])
                    data.append({
                        "file_name": file,
                        "resume_content": content,
                        "folder_name": folder
                    })
                except Exception as e:
                    st.warning(f"Could not read {file}: {e}")

df = pd.DataFrame(data)
st.subheader("Sample Data")
st.write(df.head())

if not df.empty:
    X = df["resume_content"]
    y = df["folder_name"]

    vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    st.success(f"Model trained! Accuracy: {acc:.2f}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(VEC_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    st.info("Model and vectorizer saved.")

# Prediction section
st.subheader("Classify a New Resume (using Pickle model)")
resume_text = st.text_area("Paste resume content here:")

if st.button("Predict Category"):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VEC_PATH):
        st.error("Model not trained yet. Please upload resumes and train the model first.")
    elif resume_text.strip() == "":
        st.warning("Please enter some resume content.")
    else:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(VEC_PATH, "rb") as f:
            vectorizer = pickle.load(f)

        input_vec = vectorizer.transform([resume_text])
        prediction = model.predict(input_vec)[0]
        st.success(f"Predicted Category: **{prediction}**")
