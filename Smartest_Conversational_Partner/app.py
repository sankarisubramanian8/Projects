# streamlit_app.py
import streamlit as st
import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Your rating to sentiment mapping
def rating_to_sentiment(rating):
    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('/content/chatgpt_reviews - chatgpt_reviews.csv')  # update path if needed
    # Drop rows with missing values just in case
    df.dropna(subset=['review', 'rating'], inplace=True)
    # Create sentiment label
    df['sentiment'] = df['rating'].apply(rating_to_sentiment)
    return df

df = load_data()

# Text cleaning functions
def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)
    return text

df['cleaned_review'] = df['review'].apply(clean_text)

# Split data
X = df['cleaned_review']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build pipeline model
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=10000))
])

model.fit(X_train, y_train)

# Evaluate model performance
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
rec = recall_score(y_test, y_pred, average='weighted')

st.title("📝 Sentiment Analysis on Reviews")
st.write("Model performance on test set:")

st.write(f"- Accuracy: {acc:.2f}")
st.write(f"- Precision: {prec:.2f}")
st.write(f"- Recall: {rec:.2f}")

st.markdown("---")
st.header("Try it yourself!")

user_text = st.text_area("Enter your review here:")

if user_text:
    cleaned_input = clean_text(user_text)
    pred = model.predict([cleaned_input])[0]
    st.success(f"Predicted Sentiment: **{pred}**")
