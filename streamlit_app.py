import streamlit as st
import pickle
import os

# Load models and vectorizer
MODEL_PATH = os.path.join('models', 'svm_model.pkl')
VECTORIZER_PATH = os.path.join('models', 'tfidf_vectorizer.pkl')
LABEL_ENCODER_PATH = os.path.join('models', 'label_encoder.pkl')

with open(MODEL_PATH, 'rb') as file:
    svm_model = pickle.load(file)
with open(VECTORIZER_PATH, 'rb') as file:
    vectorizer = pickle.load(file)
with open(LABEL_ENCODER_PATH, 'rb') as file:
    label_encoder = pickle.load(file)

# Streamlit UI
st.title("Sentiment Analysis")
st.write("Enter a review and analyze its sentiment")

review = st.text_area("Review")

if st.button("Analyze"):
    if review:
        review_tfidf = vectorizer.transform([review])
        predicted_label = svm_model.predict(review_tfidf)
        predicted_label_name = label_encoder.inverse_transform(predicted_label)[0]

        sentiment = 'Positive' if predicted_label_name == 'Positive' else 'Negative'
        image_path = 'static/pos.png' if predicted_label_name == 'Positive' else 'static/neg.png'

        st.write(f"Sentiment: {sentiment}")
        st.image(image_path, caption=sentiment)
    else:
        st.write("Please enter a review to analyze.")
