from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from textblob import TextBlob
import re
import json
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)


df = pd.read_csv("amazon.csv")

def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_sentiment(text):
    if pd.isna(text) or text == "":
        return 0, 0
    
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    return polarity, subjectivity

def categorize_sentiment(polarity):
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

df['clean_review_title'] = df['review_title'].apply(preprocess_text)
df['clean_review_content'] = df['review_content'].apply(preprocess_text)

title_sentiments = df['clean_review_title'].apply(get_sentiment)
df['title_polarity'] = [s[0] for s in title_sentiments]
df['title_subjectivity'] = [s[1] for s in title_sentiments]
df['title_sentiment_category'] = df['title_polarity'].apply(categorize_sentiment)

content_sentiments = df['clean_review_content'].apply(get_sentiment)
df['content_polarity'] = [s[0] for s in content_sentiments]
df['content_subjectivity'] = [s[1] for s in content_sentiments]
df['content_sentiment_category'] = df['content_polarity'].apply(categorize_sentiment)

df['rating_clean'] = pd.to_numeric(df['rating'].astype(str).str.replace('|', ''), errors='coerce')

product_names = df['product_name'].unique().tolist()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search_products')
def search_products():
    query = request.args.get('q', '').lower()
    if not query:
        return jsonify([])
    
    matching_products = [name for name in product_names if query in name.lower()]
    return jsonify(matching_products[:10])  

@app.route('/analyze', methods=['POST'])
def analyze():
    product_name = request.form.get('product_name')
    
    if not product_name:
        return jsonify({'error': 'Product name is required'})

    product_data = df[df['product_name'] == product_name]
    
    if product_data.empty:
        return jsonify({'error': 'Product not found'})

    title_sentiment_counts = product_data['title_sentiment_category'].value_counts().to_dict()
    content_sentiment_counts = product_data['content_sentiment_category'].value_counts().to_dict()
    
    avg_title_polarity = product_data['title_polarity'].mean()
    avg_content_polarity = product_data['content_polarity'].mean()
    avg_rating = product_data['rating_clean'].mean()
    
    sample_reviews = product_data[['review_title', 'review_content', 'content_polarity', 'content_sentiment_category']].head(5).to_dict('records')
    
    response = {
        'product_name': product_name,
        'total_reviews': len(product_data),
        'title_sentiment_counts': title_sentiment_counts,
       'content_sentiment_counts': content_sentiment_counts,
        'avg_title_polarity': round(avg_title_polarity, 3),
        'avg_content_polarity': round(avg_content_polarity, 3),
        'avg_rating': round(avg_rating, 2) if not pd.isna(avg_rating) else 'N/A',
        'sample_reviews': sample_reviews,
        'category': product_data['category'].iloc[0] if not product_data.empty else 'N/A'
    }
    
    return jsonify(response)

@app.route('/dashboard')
def dashboard():

    total_reviews = len(df)
    title_sentiment_counts = df['title_sentiment_category'].value_counts().to_dict()
    content_sentiment_counts = df['content_sentiment_category'].value_counts().to_dict()
    
    overall_stats = {
        'total_reviews': total_reviews,
        'title_sentiment_counts': title_sentiment_counts,
        'content_sentiment_counts': content_sentiment_counts,
        'avg_title_polarity': round(df['title_polarity'].mean(), 3),
        'avg_content_polarity': round(df['content_polarity'].mean(), 3),
        'avg_rating': round(df['rating_clean'].mean(), 2),
        'rating_sentiment_corr': round(df['rating_clean'].corr(df['content_polarity']), 3)
    }
    
    return render_template('dashboard.html', stats=overall_stats)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
