import streamlit as st
import pandas as pd
import requests
import time
import os
import re
import json
import warnings
import numpy as np
from bs4 import BeautifulSoup
from joblib import load
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import nltk
import textstat

DATA_DIR = 'data'
MODELS_DIR = 'models'

EXTRACTED_CONTENT_PATH = os.path.join(DATA_DIR, 'extracted_content.csv')
MODEL_PATH = os.path.join(MODELS_DIR, 'quality_model.pkl')
VECTORIZER_PATH = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')

SIMILARITY_THRESHOLD = 0.80
THIN_CONTENT_WORD_COUNT = 500

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab/english')
except nltk.downloader.DownloadError:
    nltk.download('punkt_tab', quiet=True)


# Utility Functions

@st.cache_data
def parse_html_content(html_content):
    """Parses HTML to extract title, body text, and word count."""
    if pd.isna(html_content) or not html_content:
        return 'No Title', '', 0
    
    try:
        soup = BeautifulSoup(str(html_content), 'html.parser')
        title = soup.title.string.strip() if soup.title and soup.title.string else 'No Title'
        
        main_content_div = soup.find('main') or soup.find('article') or soup.find('body')
        
        if main_content_div:
            body_text = main_content_div.get_text(separator=' ', strip=True)
        else:
            body_text = soup.body.get_text(separator=' ', strip=True) if soup.body else ''

        clean_text = re.sub(r'\s+', ' ', body_text).strip()
        word_count = len(clean_text.split())

        return title, clean_text, word_count
        
    except Exception:
        return 'Parsing Error', '', 0 

@st.cache_data
def calculate_text_features(text):
    """Calculates sentence count and Flesch Reading Ease score, returns tuple."""
    if not text:
        return (0, 0.0)
        
    clean_text = str(text).lower()
    
    try:
        sentence_count = len(sent_tokenize(clean_text))
    except LookupError:
        sentence_count = 0 # Fallback if NLTK data is missing
    
    try:
        flesch_score = textstat.flesch_reading_ease(clean_text)
    except:
        flesch_score = 0.0 
        
    return (sentence_count, flesch_score)

# Data/Model Loading

@st.cache_resource
def load_assets():
    """Loads the trained model, vectorizer, and existing corpus data."""
    try:
        model = load(MODEL_PATH)
        vectorizer = load(VECTORIZER_PATH)
        existing_df = pd.read_csv(EXTRACTED_CONTENT_PATH)
        return model, vectorizer, existing_df
    except FileNotFoundError as e:
        st.error(f"Required file not found: {e}. Please ensure 'models/' and 'data/' directories are structured correctly.")
        return None, None, None

model, vectorizer, extracted_df = load_assets()

#  Core Analysis Logic

@st.cache_data
def realtime_scrape_url(url):
    """Scrapes a single URL, bypassing SSL verification for robustness."""
    warnings.filterwarnings("ignore", category=requests.packages.urllib3.exceptions.InsecureRequestWarning)
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; SEOContentAnalyzer/1.0)'} 
        time.sleep(1.0) 
        response = requests.get(url, headers=headers, timeout=10, verify=False) 
        response.raise_for_status() 
        return response.text
    except requests.exceptions.RequestException as e: 
        st.error(f"Scraping failed for {url}. Status: {e}")
        return None

def analyze_url(url, model, vectorizer, existing_df):
    """Performs end-to-end content analysis on a single URL."""
    if model is None or vectorizer is None or existing_df is None:
        return {"error": "Pipeline assets could not be loaded."}

    # Scraping 
    html_content = realtime_scrape_url(url)
    if html_content is None:
        return {"url": url, "error": "Scraping failed or received bad status code."}

    # Parsing and Feature Extraction
    title, body_text, word_count = parse_html_content(html_content)
    if not body_text:
        return {"url": url, "error": "Parsing failed or content is empty."}
        
    sentence_count, readability = calculate_text_features(body_text)

    # Model Prediction Preparation
    X_core_features = pd.DataFrame({
        'word_count': [word_count],
        'sentence_count': [sentence_count],
        'flesch_reading_ease': [readability]
    })
    
    # TF-IDF vector
    new_tfidf_vector = vectorizer.transform([body_text])
    
    # Combine and ensure string column names for prediction
    X_predict = pd.concat([X_core_features.reset_index(drop=True), 
                           pd.DataFrame(new_tfidf_vector.toarray()).reset_index(drop=True)], axis=1)
    X_predict.columns = X_predict.columns.astype(str)

    # Quality Score
    quality_label = model.predict(X_predict)[0]
    
    # Duplicate Check
    corpus_content_df = existing_df[existing_df['body_text'].str.len() > 0].copy()
    corpus_tfidf = vectorizer.transform(corpus_content_df['body_text'])
    corpus_urls = corpus_content_df['url']
    
    new_sims = cosine_similarity(new_tfidf_vector, corpus_tfidf)[0]
    
    similar_to = []
    for i, sim in enumerate(new_sims):
        if sim >= SIMILARITY_THRESHOLD and sim < 0.999: 
            similar_to.append({
                "url": corpus_urls.iloc[i], 
                "similarity": round(sim, 4)
            })

    # Final Output
    return {
        "title": title,
        "url": url,
        "word_count": word_count,
        "readability": round(readability, 1),
        "quality_label": quality_label,
        "is_thin": bool(word_count < THIN_CONTENT_WORD_COUNT),
        "similar_to": similar_to
    }


# Streamlit UI

def main():
    st.set_page_config(page_title="SEO Content Detector", layout="wide")
    st.title("SEO Content Quality & Duplicate Detector")
    st.markdown("Enter a URL to analyze its quality, readability, and check for duplicates against the trained corpus.")

    url_input = st.text_input("Enter URL to Analyze:", "https://en.wikipedia.org/wiki/Search_engine_optimization")
    
    if st.button("Analyze Content") and url_input:
        
        if model is None:
            st.error("Cannot proceed. Model and data files are required.")
            return

        with st.spinner(f"Scraping and analyzing {url_input}... This may take a moment."):
            analysis_result = analyze_url(url_input, model, vectorizer, extracted_df)

        st.subheader("Analysis Results")
        
        if "error" in analysis_result:
            st.error(analysis_result["error"])
            return

        # Display Core Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Word Count", analysis_result['word_count'])
        col2.metric("Readability (Flesch)", f"{analysis_result['readability']}", help="Higher is easier to read (Score of 60-70 is standard).")
        col3.metric("ML Quality Score", analysis_result['quality_label'])

        st.markdown(f"**Page Title:** *{analysis_result['title']}*")
        
        # Display Thin Content Status
        if analysis_result['is_thin']:
            st.warning(f" **THIN CONTENT ALERT:** Word count is below the {THIN_CONTENT_WORD_COUNT} threshold.")
        else:
            st.success(" Content is NOT classified as thin.")

        # Display Duplicates
        st.markdown("---")
        st.subheader("Duplicate/Highly Similar Content Check")

        if analysis_result['similar_to']:
            st.error(f" **{len(analysis_result['similar_to'])} Highly Similar Page(s) Found in Corpus!**")
            
            sim_df = pd.DataFrame(analysis_result['similar_to'])
            sim_df.columns = ["Similar URL", "Similarity Score"]
            st.dataframe(sim_df, use_container_width=True)
            st.caption(f"Threshold: {SIMILARITY_THRESHOLD * 100:.0f}%")
        else:
            st.info("No highly similar content found in the training corpus.")
            
if __name__ == "__main__":
    main()
