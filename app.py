import os
import torch
import numpy as np
import pickle
import requests
from bs4 import BeautifulSoup
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import hashlib
import time
from cachetools import TTLCache
from concurrent.futures import ThreadPoolExecutor
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load FinBERT model and tokenizer
finbert_model_dir = './saved_finbert_model'
finbert = BertForSequenceClassification.from_pretrained(finbert_model_dir)
tokenizer = BertTokenizer.from_pretrained(finbert_model_dir)

# Load Random Forest model and metadata
random_forest_model_dir = './saved_best_model'
model_save_path = os.path.join(random_forest_model_dir, 'random_forest_model.pkl')
metadata_save_path = os.path.join(random_forest_model_dir, 'metadata_scaler.pkl')

# Load Random Forest model and scaler
with open(model_save_path, 'rb') as f:
    random_forest_model = pickle.load(f)

with open(metadata_save_path, 'rb') as f:
    scaler = pickle.load(f)

# Define function to get FinBERT predictions
def get_finbert_predictions(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = finbert(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1).numpy()
        predictions = np.argmax(probabilities, axis=1)
    return predictions, probabilities

# Caching setup to store previously processed URLs
cache = TTLCache(maxsize=100, ttl=3600)  # Cache up to 100 items for 1 hour
executor = ThreadPoolExecutor(max_workers=5)

# Scrape metadata from URL (asynchronous with caching)
def scrape_metadata(url):
    url_hash = hashlib.md5(url.encode()).hexdigest()
    if url_hash in cache:
        return cache[url_hash]

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            # Example scraping logic (can be customized based on website structure)
            author = soup.find('meta', attrs={'name': 'author'})
            author = author['content'] if author else 'Unknown'
            
            tags = [tag['content'] for tag in soup.find_all('meta', attrs={'name': 'keywords'})]
            tags = tags if tags else ['None']

            has_image = 1 if soup.find('img') else 0
            site_url = url.split('/')[2]

            published_date = soup.find('meta', attrs={'property': 'article:published_time'})
            published_date = published_date['content'] if published_date else None

            if published_date:
                # Convert to datetime and calculate age
                published_date = pd.to_datetime(published_date, errors='coerce')
                days_since_published = (pd.Timestamp.now() - published_date).days
            else:
                days_since_published = 0

            # Add more sophisticated metadata features
            language = detect(soup.get_text())  # Detect the language of the page

            metadata_features = [[
                author,
                len(tags),
                has_image,
                days_since_published,
                site_url,
                language
            ]]
            scaled_features = scaler.transform(metadata_features)
            cache[url_hash] = scaled_features
            return scaled_features
        else:
            st.error("Failed to fetch the URL content.")
            return None
    except Exception as e:
        st.error(f"Error while scraping metadata: {e}")
        return None

# Combined prediction function with Random Forest
def combined_prediction(text, url=None):
    finbert_pred, finbert_probs = get_finbert_predictions([text])
    finbert_confidence = finbert_probs[0][finbert_pred[0]]

    # Give priority to FinBERT prediction
    if finbert_confidence > 0.7:  # Threshold to prioritize FinBERT
        return "Real" if finbert_pred[0] == 1 else "Fake", finbert_confidence

    if url:
        future = executor.submit(scrape_metadata, url)
        metadata_features = future.result()
        if metadata_features is not None:
            # Use Random Forest model for metadata prediction
            random_forest_pred = random_forest_model.predict(metadata_features)
            random_forest_confidence = random_forest_model.predict_proba(metadata_features).max()
            return "Real" if random_forest_pred[0] == 1 else "Fake", random_forest_confidence
    else:
        return "Real" if finbert_pred[0] == 1 else "Fake", finbert_confidence

# Streamlit UI
def main():
    st.title("Financial News Authenticity Predictor")
    st.write("Enter a text or a URL to predict if the news is Real or Fake.")

    option = st.selectbox("Choose input type:", ("Text", "URL"))

    if option == "Text":
        text = st.text_area("Enter the news text:")
        if st.button("Predict"):
            if text:
                result, confidence = combined_prediction(text)
                st.write(f"The text is predicted to be: {result} (Confidence: {confidence:.2f})")
                # Generate word cloud
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(plt)

                # User feedback
                feedback = st.radio("Was this prediction correct?", ("Yes", "No"))
                if feedback:
                    st.write("Thank you for your feedback!")
            else:
                st.error("Please enter some text.")

    elif option == "URL":
        url = st.text_input("Enter the news URL:")
        if st.button("Predict"):
            if url:
                with st.spinner('Fetching and analyzing URL...'):
                    start_time = time.time()
                    result, confidence = combined_prediction(text=None, url=url)
                    elapsed_time = time.time() - start_time
                    st.write(f"The news is predicted to be: {result} (Confidence: {confidence:.2f})")
                    st.write(f"Prediction took {elapsed_time:.2f} seconds.")

                    # Display metadata extracted
                    metadata_features = scrape_metadata(url)
                    if metadata_features is not None:
                        st.write(f"Extracted Metadata: {metadata_features}")

                    # User feedback
                    feedback = st.radio("Was this prediction correct?", ("Yes", "No"))
                    if feedback:
                        st.write("Thank you for your feedback!")
            else:
                st.error("Please enter a URL.")

    # FAQ Section
    st.sidebar.title("FAQ")
    st.sidebar.markdown("**Q: How does this model work?**")
    st.sidebar.markdown("A: This model uses FinBERT to analyze the text content and a Random Forest model to analyze metadata from URLs to predict if the news is real or fake.")
    st.sidebar.markdown("**Q: What is the confidence score?**")
    st.sidebar.markdown("A: The confidence score represents how sure the model is about its prediction.")
    st.sidebar.markdown("**Q: Can I trust these predictions?**")
    st.sidebar.markdown("A: While the model aims to be accurate, it's not perfect. Use the predictions as guidance, not as absolute truth.")

if __name__ == "__main__":
    main()

