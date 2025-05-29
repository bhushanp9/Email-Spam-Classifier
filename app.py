import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from nltk.tokenize import word_tokenize  # this is correct
# Remove: from nltk import punkt_tab

# Download required resources
nltk.download('punkt')      # for tokenization
nltk.download('stopwords')  # for stopwords

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    tokens = word_tokenize(text)  # use word_tokenize here
    filtered = [word for word in tokens if word.isalnum()]
    clean = [ps.stem(word) for word in filtered if word not in stopwords.words('english')]
    return " ".join(clean)

# Load the fitted vectorizer and model
with open("clean_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("clean_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("📩 Email/SMS Spam Classifier")

input_sms = st.text_area("Enter your message:")

if st.button("Predict"):
    if input_sms.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = vectorizer.transform([transformed_sms])
        prediction = model.predict(vector_input)[0]

        if prediction == 1:
            st.error("🚫 Spam Detected")
        else:
            st.success("✅ Not Spam")
