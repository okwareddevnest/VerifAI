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
        """Analyze article for bias and generate comprehensive results"""
        if not article_data:
            return self._get_default_results()
        
        try:
            # Get article text
            article_text = article_data.get('text', '')
            if not article_text:
                return self._get_default_results()
            
            # Perform multi-factor analysis
            sentiment_scores = self.sia.polarity_scores(article_text)
            language_patterns = self._analyze_language_patterns(article_text)
            source_analysis = self._analyze_sources(article_data, related_articles)
            fact_checks = self._perform_fact_checking(article_text)
            historical_context = self._get_historical_context(article_text)
            
            # Calculate weighted bias score
            bias_components = {
                'sentiment': self._normalize_sentiment(sentiment_scores['compound']),
                'language': language_patterns['bias_score'],
                'source': source_analysis['bias_score'],
                'factual': fact_checks['bias_score']
            }
            
            # Weighted average for final bias score
            weights = {
                'sentiment': 0.2,
                'language': 0.3,
                'source': 0.3,
                'factual': 0.2
            }
            
            final_bias_score = sum(
                score * weights[component]
                for component, score in bias_components.items()
            )
            
            # Determine political leaning
            political_leaning = self._get_political_leaning(
                final_bias_score,
                language_patterns['partisan_terms']
            )
            
            # Generate key findings
            key_findings = self._generate_key_findings(
                bias_components,
                language_patterns,
                source_analysis,
                fact_checks
            )
            
            return {
                'bias_score': final_bias_score,
                'political_leaning': political_leaning,
                'sentiment_scores': sentiment_scores,
                'sentiment': self._get_sentiment_label(sentiment_scores['compound']),
                'bias_indicators': language_patterns['indicators'],
                'key_findings': key_findings,
                'source_analysis': {
                    'credibility': source_analysis['credibility'],
                    'fact_consistency': source_analysis['consistency'],
                    'perspective_diversity': source_analysis['diversity']
                },
                'fact_checks': fact_checks['results'],
                'historical_context': historical_context,
                'recommendations': self._generate_recommendations(
                    final_bias_score,
                    source_analysis,
                    fact_checks
                )
            }
            
        except Exception as e:
            print(f"Error in bias analysis: {str(e)}")
            return self._get_default_results()
            
    def _normalize_sentiment(self, compound_score: float) -> float:
        """Convert VADER compound score to bias score range"""
        return compound_score * 0.5  # Scale to [-0.5, 0.5] range
        
    def _get_political_leaning(self, bias_score: float, partisan_terms: Dict) -> str:
        """Determine political leaning based on bias score and language"""
        # Base category on bias score
        if bias_score <= -0.6:
            base_category = "far_left"
        elif bias_score <= -0.2:
            base_category = "left"
        elif bias_score <= 0.2:
            base_category = "center"
        elif bias_score <= 0.6:
            base_category = "right"
        else:
            base_category = "far_right"
            
        # Adjust based on partisan language
        left_terms = sum(partisan_terms['left'].values())
        right_terms = sum(partisan_terms['right'].values())
        
        if abs(left_terms - right_terms) > 5:
            if left_terms > right_terms:
                return "far_left" if base_category == "left" else "left"
            else:
                return "far_right" if base_category == "right" else "right"
                
        return base_category
        
    def _generate_key_findings(self, bias_components: Dict, language_patterns: Dict,
                             source_analysis: Dict, fact_checks: Dict) -> List[str]:
        """Generate key insights from analysis results"""
        findings = []
        
        # Add bias-related findings
        if abs(bias_components['sentiment']) > 0.3:
            findings.append(
                f"Strong {'negative' if bias_components['sentiment'] < 0 else 'positive'} "
                "emotional tone detected in the writing"
            )
            
        if language_patterns['loaded_words_count'] > 5:
            findings.append(
                "Significant use of loaded language and emotional terms"
            )
            
        if source_analysis['diversity'] < 0.3:
            findings.append(
                "Limited perspective diversity in source material"
            )
            
        if fact_checks['accuracy'] < 0.7:
            findings.append(
                "Several factual claims require additional verification"
            )
            
        return findings
        
    def _generate_recommendations(self, bias_score: float, source_analysis: Dict,
                                fact_checks: Dict) -> List[str]:
        """Generate recommendations for balanced consumption"""
        recommendations = []
        
        if abs(bias_score) > 0.4:
            recommendations.append(
                "Consider consulting sources with different political perspectives"
            )
            
        if source_analysis['credibility'] < 0.6:
            recommendations.append(
                "Verify claims against highly credible news sources"
            )
            
        if fact_checks['accuracy'] < 0.8:
            recommendations.append(
                "Cross-reference key facts with official or primary sources"
            )
            
        return recommendations
        
    def _get_default_results(self) -> Dict:
        """Return default results structure"""
        return {
            "error": "Analysis could not be completed",
            "bias_score": 0.0,
            "political_leaning": "unknown",
            "sentiment": "neutral",
            "bias_indicators": [],
            "key_findings": [],
            "recommendations": []
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
