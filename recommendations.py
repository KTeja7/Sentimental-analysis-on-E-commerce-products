import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class ProductRecommender:
    def __init__(self, df):
        self.df = df.copy()
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.tfidf_matrix = None
        self.cosine_sim = None
        self._prepare_recommendations()
    
    def _prepare_recommendations(self):
        """Prepare TF-IDF matrix and cosine similarity for recommendations"""
        # Combine review content and titles for better similarity
        self.df['combined_text'] = (
            self.df['clean_review_title'].fillna('') + ' ' + 
            self.df['clean_review_content'].fillna('')
        )
        
        # Create TF-IDF matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['combined_text'])
        
        # Calculate cosine similarity
        self.cosine_sim = cosine_similarity(self.tfidf_matrix)
    
    def get_recommendations_by_sentiment(self, product_name, top_n=5):
        """Get product recommendations based on sentiment and similarity"""
        try:
            # Find the product
            product_data = self.df[self.df['product_name'] == product_name]
            
            if product_data.empty:
                return []
            
            # Get the average sentiment of the product
            avg_sentiment = product_data['content_polarity'].mean()
            
            # Get similar products based on content
            product_indices = product_data.index.tolist()
            
            # Calculate average similarity for each product
            product_similarities = {}
            for idx in product_indices:
                sim_scores = list(enumerate(self.cosine_sim[idx]))
                for i, score in sim_scores:
                    if i not in product_indices:  # Exclude the same product
                        product_idx = self.df.iloc[i]['product_name']
                        if product_idx not in product_similarities:
                            product_similarities[product_idx] = []
                        product_similarities[product_idx].append(score)
            
            # Average similarity scores for each product
            avg_similarities = {
                product: np.mean(scores) 
                for product, scores in product_similarities.items()
            }
            
            # Get sentiment for each product
            product_sentiments = {}
            for product in avg_similarities.keys():
                product_sent = self.df[self.df['product_name'] == product]
                product_sentiments[product] = product_sent['content_polarity'].mean()
            
            # Filter products with similar sentiment (within 0.2 range)
            sentiment_range = 0.2
            similar_sentiment_products = {
                product: sentiment 
                for product, sentiment in product_sentiments.items()
                if abs(sentiment - avg_sentiment) <= sentiment_range
            }
            
            # Combine similarity and sentiment scores
            recommendations = []
            for product in similar_sentiment_products:
                similarity_score = avg_similarities.get(product, 0)
                sentiment_score = similar_sentiment_products[product]
                
                # Calculate recommendation score (70% similarity + 30% sentiment alignment)
                recommendation_score = (similarity_score * 0.7) + (1 - abs(sentiment_score - avg_sentiment) * 0.3)
                
                product_info = self.df[self.df['product_name'] == product].iloc[0]
                recommendations.append({
                    'product_name': product,
                    'category': product_info['category'],
                    'rating': product_info['rating_clean'],
                    'avg_sentiment': sentiment_score,
                    'similarity_score': similarity_score,
                    'recommendation_score': recommendation_score,
                    'review_count': len(self.df[self.df['product_name'] == product])
                })
            
            # Sort by recommendation score and return top N
            recommendations.sort(key=lambda x: x['recommendation_score'], reverse=True)
            
            return recommendations[:top_n]
            
        except Exception as e:
            print(f"Error getting recommendations: {e}")
            return []
    
    def get_top_products_by_sentiment(self, sentiment_type='positive', top_n=10):
        """Get top products by sentiment type"""
        if sentiment_type == 'positive':
            top_products = self.df.groupby('product_name')['content_polarity'].mean().sort_values(ascending=False).head(top_n)
        elif sentiment_type == 'negative':
            top_products = self.df.groupby('product_name')['content_polarity'].mean().sort_values(ascending=True).head(top_n)
        else:  # neutral
            top_products = self.df.groupby('product_name')['content_polarity'].mean().sort_values(key=lambda x: abs(x - 0)).head(top_n)
        
        recommendations = []
        for product_name, avg_sentiment in top_products.items():
            product_info = self.df[self.df['product_name'] == product_name].iloc[0]
            recommendations.append({
                'product_name': product_name,
                'category': product_info['category'],
                'rating': product_info['rating_clean'],
                'avg_sentiment': avg_sentiment,
                'review_count': len(self.df[self.df['product_name'] == product_name])
            })
        
        return recommendations
    
    def save_recommendations_to_file(self, filename='product_recommendations.csv'):
        """Save recommendations to a separate file"""
        # Get top positive products
        positive_products = self.get_top_products_by_sentiment('positive', 20)
        
        # Create recommendations DataFrame
        recommendations_df = pd.DataFrame(positive_products)
        
        # Add recommendation type
        recommendations_df['recommendation_type'] = 'Top Positive'
        
        # Save to file
        recommendations_df.to_csv(filename, index=False)
        print(f"Recommendations saved to {filename}")
        
        return recommendations_df

def generate_recommendations(df):
    """Generate and save product recommendations"""
    recommender = ProductRecommender(df)
    
    # Save general recommendations
    recommendations_df = recommender.save_recommendations_to_file()
    
    return recommender

if __name__ == "__main__":
    # Load data and generate recommendations
    df = pd.read_csv("amazon.csv")
    
    # Preprocess (same as in app.py)
    import re
    def preprocess_text(text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    from textblob import TextBlob
    
    def get_sentiment(text):
        if pd.isna(text) or text == "":
            return 0, 0
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        return polarity, subjectivity
    
    df['clean_review_title'] = df['review_title'].apply(preprocess_text)
    df['clean_review_content'] = df['review_content'].apply(preprocess_text)
    df['rating_clean'] = pd.to_numeric(df['rating'].astype(str).str.replace('|', ''), errors='coerce')
    
    # Apply sentiment analysis
    content_sentiments = df['clean_review_content'].apply(get_sentiment)
    df['content_polarity'] = [s[0] for s in content_sentiments]
    
    # Generate recommendations
    recommender = generate_recommendations(df)
    print("Recommendations generated successfully!")
