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
import json
import time
from sentence_transformers import SentenceTransformer

class NewsSearcher:
    def __init__(self):
        """Initialize the news searcher with embedding model and Snowflake connection"""
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.snowflake = SnowflakeManager()
    
    def find_related(self, article_data: Dict, top_k: int = 5) -> List[Dict]:
        """Find related articles using semantic search"""
        try:
            # Extract text from article data
            text = article_data.get('text', '')
            if not text:
                return []
            
            # Search for related articles
            results = self.snowflake.semantic_search(text, top_k=top_k)
            
            # Add credibility scores
            for result in results:
                result['credibility_score'] = self._calculate_credibility(result)
                result['final_score'] = (
                    result['score'] * 0.7 +  # Semantic similarity
                    result['credibility_score'] * 0.3  # Source credibility
                )
            
            # Sort by final score
            results.sort(key=lambda x: x['final_score'], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            print(f"Error finding related articles: {str(e)}")
            return []
    
    def _calculate_credibility(self, article: Dict) -> float:
        """Calculate credibility score for an article"""
        # This would ideally check against a database of known source credibility
        # For now, return a default score
        return 0.7
    
    def __del__(self):
        """Cleanup resources"""
        try:
            self.snowflake.close()
        except:
            pass
