# app.py

import streamlit as st
import re
import nltk
import pandas as pd
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

# Download NLTK data
nltk.download('punkt')

# Load NLTK resources
nltk.download('stopwords')

# Load the CountVectorizer
cv = CountVectorizer(max_features=4500)

# Load the model from file
with open('cv.pkl', 'rb') as file:
    cv = pickle.load(file)
with open('spam_model.pkl', 'rb') as file:
    spam_detect_model = pickle.load(file)

# Function to preprocess and predict
def predict_spam_or_ham(input_text):
    ps = PorterStemmer()
    review = re.sub('[^a-zA-Z]', ' ', input_text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    input_vector = cv.transform([review]).toarray()
    prediction = spam_detect_model.predict(input_vector)
    return prediction

# Main Streamlit app
def main():
    st.set_page_config(page_title="Spam or Ham Classifier", layout="wide")
    
    # Header
    st.title("Spam or Ham Classifier")
    st.write("Welcome to the Spam or Ham Classifier. Enter text below to determine if it's spam or ham.")
    
    # Input field
    user_input = st.text_area("Enter your message here", height=200)
    
    # Predict button
    if st.button("Predict"):
        if user_input.strip():
            prediction = predict_spam_or_ham(user_input)
            if prediction == 1:
                st.success("Prediction: SPAM")
            else:
                st.success("Prediction: HAM")
        else:
            st.warning("Please enter some text to predict.")
    
    # About section
    st.sidebar.title("About")
    st.sidebar.info(
        "This app predicts whether a given text message is spam or ham (not spam). "
        "It uses a machine learning model trained on SMS messages data."
    )
    
    # Display dataset statistics
    st.sidebar.title("Dataset Statistics")
    st.sidebar.write("Total Messages: 5574")
    st.sidebar.write("Spam Messages: 747")
    st.sidebar.write("Ham Messages: 4827")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.write("Developed by [Your Name]")
    st.sidebar.write("Find the code on [GitHub](https://github.com/yourusername/your-repo)")

if __name__ == "__main__":
    main()
