from typing import Dict, List, Tuple
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
import json
import os
from .snowflake_integration import SnowflakeManager

class BiasDetector:
    def __init__(self):
        # Download required NLTK data
        nltk.download('vader_lexicon')
        
        # Initialize sentiment analyzer
        self.sia = SentimentIntensityAnalyzer()
        
        # Initialize Snowflake manager
        self.snowflake = SnowflakeManager()
        
        # Load bias indicators
        self._load_bias_indicators()
        
        # Define system prompt for Mistral
        self.system_prompt = """You are an expert in media bias analysis. Analyze the given article and related sources for bias.
        Provide a structured analysis in JSON format with the following fields:
        {
            "bias_score": float (-1.0 to 1.0, where -1 is far left and 1 is far right),
            "confidence": float (0.0 to 1.0),
            "key_findings": list of strings,
            "bias_indicators": {
                "language_patterns": list of strings,
                "source_credibility": string,
                "fact_consistency": string
            },
            "recommendations": list of strings
        }
        
        Base your analysis on:
        1. Language patterns and tone
        2. Source credibility and diversity
        3. Fact consistency across sources
        4. Historical context and perspective
        """
    
    def _load_bias_indicators(self):
        """Load bias indicators from a predefined list"""
        self.bias_indicators = {
            'emotional_language': [
                'outrageous', 'shocking', 'horrible', 'terrible',
                'amazing', 'incredible', 'wonderful', 'fantastic'
            ],
            'loaded_words': [
                'radical', 'extremist', 'terrorist', 'socialist',
                'conspiracy', 'propaganda', 'regime', 'elite'
            ],
            'partisan_terms': [
                'leftist', 'rightist', 'liberal', 'conservative',
                'democrat', 'republican', 'progressive', 'traditionalist'
            ]
        }
    
    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze the sentiment of the text"""
        scores = self.sia.polarity_scores(text)
        
        # Determine overall sentiment
        if scores['compound'] >= 0.05:
            sentiment = 'positive'
        elif scores['compound'] <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'scores': scores
        }
    
    def _analyze_language_patterns(self, text: str) -> Dict:
        """Analyze language patterns for bias indicators"""
        text_lower = text.lower()
        sentences = sent_tokenize(text)
        
        # Count bias indicators
        indicator_counts = {
            category: sum(1 for word in words if word in text_lower)
            for category, words in self.bias_indicators.items()
        }
        
        # Calculate emotional language score
        emotional_words = sum(indicator_counts.values())
        total_words = len(text.split())
        emotional_score = emotional_words / total_words if total_words > 0 else 0
        
        return {
            'indicator_counts': indicator_counts,
            'emotional_score': emotional_score
        }
    
    def analyze(self, article_data: Dict, related_articles: List[Dict] = None) -> Dict:
        """Analyze article for bias and generate results"""
        if not article_data:
            return {
                "error": "No article data provided",
                "bias_score": 0.5,
                "political_leaning": "unknown",
                "sentiment": "neutral"
            }
        
        try:
            # Get article text
            article_text = article_data.get('text', '')
            if not article_text:
                return {
                    "error": "No article text found",
                    "bias_score": 0.5,
                    "political_leaning": "unknown",
                    "sentiment": "neutral"
                }
            
            # Perform sentiment analysis
            sentiment_scores = self.sia.polarity_scores(article_text)
            
            # Get base analysis from Snowflake
            llm_results = self.snowflake.generate_analysis(
                article_text=article_text,
                related_articles=related_articles or []
            )
            
            # Combine results
            analysis_results = {
                **llm_results,
                "sentiment_scores": sentiment_scores,
                "sentiment": self._get_sentiment_label(sentiment_scores['compound'])
            }
            
            return analysis_results
            
        except Exception as e:
            print(f"Error in bias analysis: {str(e)}")
            return {
                "error": str(e),
                "bias_score": 0.5,
                "political_leaning": "unknown",
                "sentiment": "neutral"
            }
    
    def _get_sentiment_label(self, compound_score: float) -> str:
        """Convert compound sentiment score to a label"""
        if compound_score >= 0.05:
            return "positive"
        elif compound_score <= -0.05:
            return "negative"
        else:
            return "neutral"
    
    def __del__(self):
        """Cleanup Snowflake resources"""
        try:
            self.snowflake.close()
        except:
            pass
