import os
from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import warnings
from recommendations import ProductRecommender
import io
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'super_secret_sentiment_key'
os.makedirs('uploads', exist_ok=True)
app.config['UPLOAD_FOLDER'] = 'uploads'

vader_analyzer = SentimentIntensityAnalyzer()

df = None
product_names = []
recommender = None

def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_sentiment(text):
    if pd.isna(text) or text == "":
        return 0.0, 0.5
    
    scores = vader_analyzer.polarity_scores(text)
    return scores['compound'], 0.5 

def categorize_sentiment(polarity):
    # Use standard VADER thresholds
    if polarity >= 0.05:
        return "Positive"
    elif polarity <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def load_and_process_data(filepath_or_buffer):
    global df, product_names, recommender
    
    try:
        new_df = pd.read_csv(filepath_or_buffer)
    except Exception as e:
        try:
            # Fallback for encoding
            new_df = pd.read_csv(filepath_or_buffer, encoding='cp1252')
        except Exception as e2:
            print(f"Error reading dataset: {e2}")
            return False
            
    # Validation / Defaulting
    if 'product_name' not in new_df.columns:
        new_df['product_name'] = 'Unknown Product'
    if 'review_title' not in new_df.columns:
        new_df['review_title'] = ''
    if 'review_content' not in new_df.columns:
        new_df['review_content'] = new_df.get('text', new_df.get('review', ''))
    if 'rating' not in new_df.columns:
        new_df['rating'] = '3.0'
    if 'category' not in new_df.columns:
        new_df['category'] = 'Unknown Category'

    df = new_df

    df['clean_review_title'] = df['review_title'].apply(preprocess_text)
    df['clean_review_content'] = df['review_content'].apply(preprocess_text)

    # Calculate sentiment via VADER
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
    
    try:
        recommender = ProductRecommender(df)
        recommender.save_recommendations_to_file()
    except Exception as e:
        print(f"Error generating recommendations: {e}")
        # Even if recommender fails, we return True so dashboard can still work
        
    return True

# Initialize default dataset
try:
    load_and_process_data("amazon.csv")
except Exception as e:
    print("Initial load failed, will proceed empty")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
        
    if file and file.filename.endswith('.csv'):
        filename = re.sub(r'[^a-zA-Z0-9_\-\.]', '', file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        success = load_and_process_data(filepath)
        if success:
            return jsonify({'message': 'Data loaded correctly! Continue to use the app.'})
        else:
            return jsonify({'error': 'Failed to process CSV data.'})
    
    return jsonify({'error': 'Invalid file format. Please upload a .csv file.'})

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
    
    # Filter data for the selected product
    product_data = df[df['product_name'] == product_name]
    
    if product_data.empty:
        return jsonify({'error': 'Product not found'})
    
    # Calculate sentiment statistics
    title_sentiment_counts = product_data['title_sentiment_category'].value_counts().to_dict()
    content_sentiment_counts = product_data['content_sentiment_category'].value_counts().to_dict()
    
    # Calculate averages
    avg_title_polarity = product_data['title_polarity'].mean()
    avg_content_polarity = product_data['content_polarity'].mean()
    avg_rating = product_data['rating_clean'].mean()
    
    # Get recommendations for this product
    try:
        recommendations = recommender.get_recommendations_by_sentiment(product_name, top_n=5)
    except Exception as e:
        print(f"Error getting recommendations: {e}")
        recommendations = []
    
    # Get sample reviews
    sample_reviews = product_data[['review_title', 'review_content', 'content_polarity', 'content_sentiment_category']].head(5).to_dict('records')
    
    # Prepare response data
    response = {
        'product_name': product_name,
        'total_reviews': len(product_data),
        'title_sentiment_counts': title_sentiment_counts,
        'content_sentiment_counts': content_sentiment_counts,
        'avg_title_polarity': round(avg_title_polarity, 3),
        'avg_content_polarity': round(avg_content_polarity, 3),
        'avg_rating': round(avg_rating, 2) if not pd.isna(avg_rating) else 'N/A',
        'sample_reviews': sample_reviews,
        'category': product_data['category'].iloc[0] if not product_data.empty else 'N/A',
        'recommendations': recommendations
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

@app.route('/get_recommendations')
def get_recommendations():
    """Get top recommended products"""
    try:
        # Get top positive products
        top_products = recommender.get_top_products_by_sentiment('positive', 10)
        return jsonify({'recommendations': top_products})
    except Exception as e:
        print(f"Error getting recommendations: {e}")
        return jsonify({'recommendations': []})

@app.route('/download_results/<product_name>')
def download_results(product_name):
    """Download analysis results as CSV"""
    try:
        # Filter data for selected product
        product_data = df[df['product_name'] == product_name]
        
        if product_data.empty:
            return jsonify({'error': 'Product not found'})
        
        # Prepare analysis results
        results_df = product_data[[
            'product_id', 'product_name', 'category', 'rating', 
            'review_title', 'review_content',
            'title_polarity', 'title_sentiment_category',
            'content_polarity', 'content_sentiment_category'
        ]].copy()
        
        # Add recommendations
        recommendations = recommender.get_recommendations_by_sentiment(product_name, top_n=5)
        
        # Create a summary section
        summary_data = {
            'Product Name': [product_name],
            'Total Reviews': [len(product_data)],
            'Average Rating': [product_data['rating_clean'].mean()],
            'Average Title Polarity': [product_data['title_polarity'].mean()],
            'Average Content Polarity': [product_data['content_polarity'].mean()],
            'Positive Reviews (%)': [
                (product_data['content_sentiment_category'].value_counts().get('Positive', 0) / len(product_data) * 100)
            ],
            'Neutral Reviews (%)': [
                (product_data['content_sentiment_category'].value_counts().get('Neutral', 0) / len(product_data) * 100)
            ],
            'Negative Reviews (%)': [
                (product_data['content_sentiment_category'].value_counts().get('Negative', 0) / len(product_data) * 100)
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        
        # Create recommendations DataFrame
        if recommendations:
            rec_df = pd.DataFrame(recommendations)
            rec_df = rec_df[['product_name', 'category', 'rating', 'avg_sentiment', 'recommendation_score']]
            rec_df.columns = ['Recommended Product', 'Category', 'Rating', 'Avg Sentiment', 'Recommendation Score']
        else:
            rec_df = pd.DataFrame(columns=['Recommended Product', 'Category', 'Rating', 'Avg Sentiment', 'Recommendation Score'])
        
        # Create CSV in memory
        output = io.StringIO()
        
        # Write summary
        output.write("PRODUCT ANALYSIS SUMMARY\n")
        summary_df.to_csv(output, index=False)
        output.write("\n\n")
        
        # Write detailed reviews
        output.write("DETAILED REVIEW ANALYSIS\n")
        results_df.to_csv(output, index=False)
        output.write("\n\n")
        
        # Write recommendations
        output.write("RECOMMENDED PRODUCTS\n")
        rec_df.to_csv(output, index=False)
        
        # Create file in memory
        output.seek(0)
        
        # Convert to bytes for send_file
        output_bytes = io.BytesIO()
        output_bytes.write(output.getvalue().encode('utf-8'))
        output_bytes.seek(0)
        
        # Create filename
        safe_product_name = re.sub(r'[^\w\s-]', '', product_name).strip()
        safe_product_name = re.sub(r'[-\s]+', '_', safe_product_name)
        filename = f"{safe_product_name}_sentiment_analysis.csv"
        
        return send_file(
            output_bytes,
            as_attachment=True,
            download_name=filename,
            mimetype='text/csv'
        )
        
    except Exception as e:
        print(f"Error downloading results: {e}")
        return jsonify({'error': 'Failed to generate download file'})

@app.route('/download_dashboard_results')
def download_dashboard_results():
    """Download complete dashboard analysis as CSV"""
    try:
        # Overall statistics
        total_reviews = len(df)
        title_sentiment_counts = df['title_sentiment_category'].value_counts().to_dict()
        content_sentiment_counts = df['content_sentiment_category'].value_counts().to_dict()
        
        # Create summary statistics
        summary_data = {
            'Metric': [
                'Total Reviews Analyzed',
                'Average Title Polarity',
                'Average Content Polarity', 
                'Average Rating',
                'Rating-Sentiment Correlation',
                'Positive Reviews (Title)',
                'Positive Reviews (Content)',
                'Negative Reviews (Title)',
                'Negative Reviews (Content)',
                'Neutral Reviews (Title)',
                'Neutral Reviews (Content)'
            ],
            'Value': [
                total_reviews,
                round(df['title_polarity'].mean(), 3),
                round(df['content_polarity'].mean(), 3),
                round(df['rating_clean'].mean(), 2),
                round(df['rating_clean'].corr(df['content_polarity']), 3),
                title_sentiment_counts.get('Positive', 0),
                content_sentiment_counts.get('Positive', 0),
                title_sentiment_counts.get('Negative', 0),
                content_sentiment_counts.get('Negative', 0),
                title_sentiment_counts.get('Neutral', 0),
                content_sentiment_counts.get('Neutral', 0)
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        
        # Get top products by sentiment
        top_positive = recommender.get_top_products_by_sentiment('positive', 10)
        top_negative = recommender.get_top_products_by_sentiment('negative', 10)
        
        # Create DataFrames
        pos_df = pd.DataFrame(top_positive)
        neg_df = pd.DataFrame(top_negative)
        
        # Create CSV in memory
        output = io.StringIO()
        
        # Write summary
        output.write("OVERALL DASHBOARD ANALYSIS\n")
        summary_df.to_csv(output, index=False)
        output.write("\n\n")
        
        # Write top positive products
        output.write("TOP POSITIVE SENTIMENT PRODUCTS\n")
        pos_df.to_csv(output, index=False)
        output.write("\n\n")
        
        # Write top negative products
        output.write("TOP NEGATIVE SENTIMENT PRODUCTS\n")
        neg_df.to_csv(output, index=False)
        
        # Create file in memory
        output.seek(0)
        
        # Convert to bytes for send_file
        output_bytes = io.BytesIO()
        output_bytes.write(output.getvalue().encode('utf-8'))
        output_bytes.seek(0)
        
        return send_file(
            output_bytes,
            as_attachment=True,
            download_name="dashboard_sentiment_analysis.csv",
            mimetype='text/csv'
        )
        
    except Exception as e:
        print(f"Error downloading dashboard results: {e}")
        return jsonify({'error': 'Failed to generate dashboard download file'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
