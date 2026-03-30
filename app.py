# app.py
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Clothing Review Sentiment", layout="centered")
st.title("👗 Women's E-Commerce Clothing Review Sentiment Predictor")
st.write(
    "Enter a review and find out if it is positive or negative. "
    "Adjust the threshold for better balance between positive/negative predictions."
)

# ==============================
# LOAD MODEL & VECTORIZER
# ==============================
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ==============================
# USER INPUT - SINGLE REVIEW
# ==============================
st.subheader("📥 Enter Review")
review_text = st.text_area("Type your review here:")

# Sidebar threshold slider
threshold = st.sidebar.slider("Positive Prediction Threshold", 0.0, 1.0, 0.5)

# ==============================
# PREDICTION LOGIC
# ==============================
if st.button("🔍 Predict Sentiment"):
    if not review_text.strip():
        st.warning("Please enter a review first!")
    else:
        vect_input = vectorizer.transform([review_text])

        # Predict probability (works for Logistic Regression & Naive Bayes)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(vect_input)[0][1]  # Positive probability
        else:
            proba = None

        # Apply threshold
        prediction = "Positive" if proba and proba > threshold else "Negative"

        # Display prediction with emoji
        emoji = "😊" if prediction == "Positive" else "😢"
        st.success(f"Prediction: {prediction} {emoji}")

        # Show probability
        if proba is not None:
            st.info(f"Positive Probability: {proba:.2f}")

# ==============================
# OPTIONAL: WORD CLOUD VISUALIZATION
# ==============================
st.subheader("☁️ Review Word Cloud")
if review_text.strip():
    wc = WordCloud(width=600, height=300, background_color='white').generate(review_text)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

# ==============================
# OPTIONAL: BATCH PREDICTION WITH CSV
# ==============================
st.subheader("📄 Batch Prediction (CSV Upload)")
uploaded_file = st.file_uploader("Upload CSV with a 'Review' column", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'Review' in df.columns:
        vect_data = vectorizer.transform(df['Review'])
        probs = model.predict_proba(vect_data)[:, 1]
        df['Prediction'] = np.where(probs > threshold, "Positive", "Negative")
        df['Probability'] = probs
        st.dataframe(df)

        # Download button
        st.download_button(
            "Download Predictions",
            df.to_csv(index=False),
            "predictions.csv"
        )
    else:
        st.error("CSV must have a 'Review' column.")