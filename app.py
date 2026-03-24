import streamlit as st
import pickle
import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Text cleaning
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# UI
st.title("📰 Fake News Detection")

user_input = st.text_area("Enter News Text")

if st.button("Predict"):
    cleaned = clean_text(user_input)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)

    if pred[0] == 1:
        st.success("✅ Real News")
    else:
        st.error("❌ Fake News")