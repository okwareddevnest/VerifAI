from typing import Dict, List, Optional
import os
from datetime import datetime, timedelta
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from .snowflake_integration import SnowflakeManager
import requests
from bs4 import BeautifulSoup

class NewsSearcher:
    def __init__(self):
        # Initialize Snowflake manager
        self.snowflake = SnowflakeManager()
        
        # Initialize NLTK components
        nltk.download('punkt')
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(stop_words='english')
    
    def _extract_keywords(self, text: str, num_keywords: int = 5) -> List[str]:
        """Extract key terms from the article text"""
        tokens = word_tokenize(text.lower())
        tokens = [t for t in tokens if t.isalnum() and t not in self.stop_words]
        
        # Create frequency distribution
        freq_dist = nltk.FreqDist(tokens)
        return [word for word, _ in freq_dist.most_common(num_keywords)]
    
    def process_url(self, url: str) -> Optional[Dict]:
        """Process a URL to extract article content"""
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract main content (this is a simple implementation)
            article = {
                'url': url,
                'domain': url.split('/')[2],
                'title': soup.title.string if soup.title else '',
                'text': ' '.join([p.get_text() for p in soup.find_all('p')])
            }
            
            return article
            
        except Exception as e:
            print(f"Error processing URL: {str(e)}")
            return None
    
    def find_related(self, article_data: Dict, top_k: int = 5) -> List[Dict]:
        """Find related articles using semantic search"""
        # Extract keywords from article
        text = f"{article_data.get('title', '')} {article_data.get('text', '')}"
        
        # Use semantic search to find related articles
        results = self.snowflake.semantic_search(text, top_k=top_k)
        
        return results
    
    def get_fact_checking_sources(self, claim: str) -> List[Dict]:
        """Search fact-checking websites using semantic search"""
        try:
            # Perform semantic search specifically on fact-checking content
            results = self.snowflake.semantic_search(
                query=claim,
                top_k=3
            )
            
            fact_checks = []
            for result in results:
                if result['score'] > 0.7:  # Minimum relevance threshold
                    fact_checks.append({
                        'claim': claim,
                        'verdict': 'Checking...',  # Would be determined by fact-checking logic
                        'confidence': float(result['score']),
                        'source': result['metadata']['domain'],
                        'url': result['metadata']['url']
                    })
            
            return fact_checks
            
        except Exception as e:
            print(f"Error searching fact-checking sources: {str(e)}")
            return []
    
    def __del__(self):
        """Cleanup Snowflake resources"""
        try:
            self.snowflake.close()
        except:
            pass
