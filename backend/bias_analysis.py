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
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
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
        scores = self.sentiment_analyzer.polarity_scores(text)
        
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
    
    def analyze(self, article_data: Dict, related_articles: List[Dict],
                fact_check_threshold: float = 0.7) -> Dict:
        """Perform comprehensive bias analysis"""
        text = article_data['text']
        
        # Analyze sentiment
        sentiment_results = self._analyze_sentiment(text)
        
        # Analyze language patterns
        language_results = self._analyze_language_patterns(text)
        
        # Get Mistral analysis
        llm_results = self.snowflake.generate_analysis(
            article_text=text,
            related_articles=related_articles,
            system_prompt=self.system_prompt
        )
        
        # Combine results
        analysis_results = {
            'bias_score': llm_results.get('bias_score', 0.0),
            'confidence': llm_results.get('confidence', 0.8),
            'sentiment': sentiment_results['sentiment'],
            'language_analysis': {
                'emotional_scores': sentiment_results['scores'],
                'style_metrics': {
                    'emotional_language': language_results['emotional_score'],
                    'loaded_words': sum(language_results['indicator_counts'].values()) / 100
                }
            },
            'bias_indicators': [
                f"Found {count} instances of {category.replace('_', ' ')}"
                for category, count in language_results['indicator_counts'].items()
                if count > 0
            ] + llm_results.get('bias_indicators', {}).get('language_patterns', []),
            'key_takeaways': llm_results.get('key_findings', []),
            'recommendations': llm_results.get('recommendations', []),
            'source_analysis': {
                'credibility': llm_results.get('bias_indicators', {}).get('source_credibility', 'Unknown'),
                'fact_consistency': llm_results.get('bias_indicators', {}).get('fact_consistency', 'Unknown')
            }
        }
        
        return analysis_results
    
    def __del__(self):
        """Cleanup Snowflake resources"""
        try:
            self.snowflake.close()
        except:
            pass
