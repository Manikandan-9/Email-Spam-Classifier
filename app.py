import streamlit as st
import pickle
from utils import preprocess_text

# Load model and vectorizer
with open("model/spam_classifier.pkl", "rb") as f:
    model, vectorizer = pickle.load(f)

# Streamlit app UI
st.title("📧 Email Spam Classifier")
st.subheader("Enter email content below to check if it's spam or ham.")

email_text = st.text_area("Email Text", height=200)

if st.button("Predict"):
    clean_text = preprocess_text(email_text)
    vector = vectorizer.transform([clean_text])
    
    # 🔍 Use probability to make prediction with custom threshold
    probs = model.predict_proba(vector)[0]
    spam_prob = probs[1]  # Probability for spam
    
    if spam_prob > 0.3:  # Custom threshold (default is 0.5)
        prediction = 'spam'
    else:
        prediction = 'ham'
    
    st.success(f"This email is classified as: **{prediction.upper()}**")
    st.write(f"Spam Probability: `{spam_prob:.2f}`")
