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
from transformers import SentenceTransformer

class NewsSearcher:
    def __init__(self):
        """Initialize the news searcher with Snowflake and Mistral"""
        self.snowflake = SnowflakeManager()
        self.mistral = MistralAPI()
        
        # Initialize embeddings model for semantic search
        self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
        
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
        """Find related articles using hybrid search (Cortex + semantic)"""
        try:
            # Extract key information
            text = article_data.get('text', '')
            
            # Generate embeddings for semantic search
            query_embedding = self.embeddings_model.encode(text)
            
            # Perform Cortex vector search
            cortex_results = self.snowflake.semantic_search(
                query_embedding,
                top_k=top_k * 2  # Get more results for reranking
            )
            
            # Use Mistral for contextual reranking
            reranked_results = self._rerank_with_mistral(text, cortex_results)
            
            # Return top K results after reranking
            return reranked_results[:top_k]
            
        except Exception as e:
            print(f"Error in find_related: {str(e)}")
            return []
            
    def _rerank_with_mistral(self, query_text: str, search_results: List[Dict]) -> List[Dict]:
        """Rerank search results using Mistral LLM"""
        try:
            # Prepare context for Mistral
            context = "\n\n".join([
                f"Title: {result['metadata']['title']}\n"
                f"Content: {result['metadata']['content'][:500]}..."
                for result in search_results
            ])
            
            # Prompt for relevance scoring
            prompt = f"""
            Given the following article text:
            {query_text[:500]}...
            
            And these potential related articles:
            {context}
            
            Score each article's relevance from 0-1 based on:
            1. Topical similarity
            2. Perspective diversity
            3. Source credibility
            4. Time relevance
            
            Return scores in JSON format.
            """
            
            # Get Mistral's evaluation
            scores = self.mistral.generate(prompt)
            
            # Parse scores and rerank results
            try:
                scores_dict = json.loads(scores)
                for i, result in enumerate(search_results):
                    result['mistral_score'] = scores_dict.get(str(i), 0.0)
                    # Combine with semantic similarity score
                    result['final_score'] = (
                        result['score'] * 0.6 +  # Semantic similarity weight
                        result['mistral_score'] * 0.4  # Mistral score weight
                    )
            except json.JSONDecodeError:
                print("Error parsing Mistral scores")
                for result in search_results:
                    result['final_score'] = result['score']
            
            # Sort by final score
            search_results.sort(key=lambda x: x['final_score'], reverse=True)
            return search_results
            
        except Exception as e:
            print(f"Error in reranking: {str(e)}")
            return search_results
            
    def test_search_quality(self, test_queries: List[str]) -> Dict:
        """Test search quality metrics"""
        results = {
            'precision': [],
            'recall': [],
            'latency': [],
            'diversity': []
        }
        
        for query in test_queries:
            start_time = time.time()
            
            # Perform search
            search_results = self.find_related({'text': query}, top_k=5)
            
            # Calculate metrics
            latency = time.time() - start_time
            
            # Calculate result diversity (source domains)
            domains = set(r['metadata'].get('domain', '') for r in search_results)
            diversity = len(domains) / len(search_results) if search_results else 0
            
            # Store metrics
            results['latency'].append(latency)
            results['diversity'].append(diversity)
            
        return {
            'avg_latency': sum(results['latency']) / len(results['latency']),
            'avg_diversity': sum(results['diversity']) / len(results['diversity']),
            'total_queries': len(test_queries)
        }
    
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
