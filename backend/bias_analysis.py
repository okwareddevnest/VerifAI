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
        words = text_lower.split()
        
        # Count bias indicators
        indicator_counts = {
            'emotional': sum(1 for word in words if word in self.bias_indicators['emotional_language']),
            'loaded': sum(1 for word in words if word in self.bias_indicators['loaded_words']),
            'partisan': sum(1 for word in words if word in self.bias_indicators['partisan_terms'])
        }
        
        # Calculate emotional language score
        total_bias_words = sum(indicator_counts.values())
        total_words = len(words)
        emotional_score = total_bias_words / total_words if total_words > 0 else 0
        
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
            
            # Perform sentiment analysis
            sentiment_scores = self.sia.polarity_scores(article_text)
            
            # Analyze language patterns
            language_analysis = self._analyze_language_patterns(article_text)
            
            # Calculate bias score components
            sentiment_bias = sentiment_scores['compound'] * 0.3  # Weight sentiment less
            language_bias = language_analysis['emotional_score'] * 0.7  # Weight language patterns more
            
            # Combine for final bias score
            bias_score = max(-1.0, min(1.0, sentiment_bias + language_bias))
            
            # Determine political leaning
            political_leaning = self._get_political_leaning(bias_score, language_analysis['indicator_counts'])
            
            # Generate key findings
            key_findings = []
            if abs(sentiment_scores['compound']) > 0.2:
                key_findings.append(
                    f"Strong {'negative' if sentiment_scores['compound'] < 0 else 'positive'} tone detected"
                )
            
            if language_analysis['emotional_score'] > 0.1:
                key_findings.append("Significant use of emotional language")
            
            # Generate recommendations
            recommendations = []
            if abs(bias_score) > 0.3:
                recommendations.append("Consider consulting sources with different perspectives")
            
            if language_analysis['emotional_score'] > 0.15:
                recommendations.append("Be aware of emotional language influence")
            
            return {
                "bias_score": bias_score,
                "political_leaning": political_leaning,
                "sentiment": self._get_sentiment_label(sentiment_scores['compound']),
                "sentiment_scores": sentiment_scores,
                "bias_indicators": language_analysis['indicator_counts'],
                "key_findings": key_findings,
                "recommendations": recommendations
            }
            
        except Exception as e:
            print(f"Error in bias analysis: {str(e)}")
            return self._get_default_results()
            
    def _normalize_sentiment(self, compound_score: float) -> float:
        """Convert VADER compound score to bias score range"""
        return compound_score * 0.5  # Scale to [-0.5, 0.5] range
        
    def _get_political_leaning(self, bias_score: float, indicator_counts: Dict) -> str:
        """Determine political leaning based on bias score and language patterns"""
        # Base category on bias score
        if bias_score <= -0.6:
            base_category = "far left"
        elif bias_score <= -0.2:
            base_category = "left"
        elif bias_score <= 0.2:
            base_category = "center"
        elif bias_score <= 0.6:
            base_category = "right"
        else:
            base_category = "far right"
        
        # Adjust based on partisan language usage
        partisan_count = indicator_counts.get('partisan', 0)
        if partisan_count > 5:  # High partisan language usage
            if bias_score < 0:
                return "far left"
            elif bias_score > 0:
                return "far right"
        
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
            "political_leaning": "center",
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
    
    def _analyze_sources(self, article_data: Dict, related_articles: List[Dict]) -> Dict:
        """Analyze sources and their credibility"""
        try:
            # Calculate source diversity
            sources = set()
            if article_data.get('domain'):
                sources.add(article_data['domain'])
            for article in (related_articles or []):
                if article.get('metadata', {}).get('domain'):
                    sources.add(article['metadata']['domain'])
            
            diversity = len(sources) / max(len(related_articles or []) + 1, 1)
            
            # Calculate source credibility
            credibility_scores = []
            for source in sources:
                # This would ideally check against a database of known source credibility
                credibility_scores.append(0.7)  # Default credibility score
            
            avg_credibility = sum(credibility_scores) / len(credibility_scores) if credibility_scores else 0.5
            
            # Calculate consistency across sources
            consistency = self._calculate_source_consistency(article_data, related_articles or [])
            
            # Calculate bias based on source analysis
            bias_score = 0.0
            if related_articles:
                left_leaning = sum(1 for a in related_articles if self._is_left_leaning(a))
                right_leaning = sum(1 for a in related_articles if self._is_right_leaning(a))
                bias_score = (right_leaning - left_leaning) / len(related_articles)
            
            return {
                'diversity': diversity,
                'credibility': avg_credibility,
                'consistency': consistency,
                'bias_score': bias_score
            }
            
        except Exception as e:
            print(f"Error in source analysis: {str(e)}")
            return {
                'diversity': 0.0,
                'credibility': 0.5,
                'consistency': 0.5,
                'bias_score': 0.0
            }
            
    def _perform_fact_checking(self, text: str) -> Dict:
        """Perform fact checking on the article text"""
        try:
            # Extract claims from text
            claims = self._extract_claims(text)
            
            # Check each claim
            results = []
            total_accuracy = 0.0
            
            for claim in claims:
                # This would ideally use a fact-checking API or database
                check_result = {
                    'claim': claim,
                    'verdict': 'Needs verification',
                    'confidence': 0.7,
                    'sources': []
                }
                results.append(check_result)
                total_accuracy += 0.7  # Default accuracy score
            
            avg_accuracy = total_accuracy / len(claims) if claims else 0.5
            
            return {
                'results': results,
                'accuracy': avg_accuracy,
                'bias_score': (1 - avg_accuracy) * 0.5  # Convert accuracy to bias score
            }
            
        except Exception as e:
            print(f"Error in fact checking: {str(e)}")
            return {
                'results': [],
                'accuracy': 0.5,
                'bias_score': 0.0
            }
            
    def _extract_claims(self, text: str) -> List[str]:
        """Extract factual claims from text"""
        sentences = sent_tokenize(text)
        claims = []
        
        # Simple claim extraction based on sentence structure
        for sentence in sentences:
            lower_sent = sentence.lower()
            # Look for sentences that make factual assertions
            if any(word in lower_sent for word in ['is', 'are', 'was', 'were', 'will', 'has', 'have']):
                claims.append(sentence)
                
        return claims[:5]  # Limit to top 5 claims
        
    def _calculate_source_consistency(self, article_data: Dict, related_articles: List[Dict]) -> float:
        """Calculate consistency of information across sources"""
        try:
            if not related_articles:
                return 0.5
                
            # Compare main points across sources
            main_text = article_data.get('text', '').lower()
            consistency_scores = []
            
            for article in related_articles:
                related_text = article.get('metadata', {}).get('content', '').lower()
                # Calculate text similarity
                similarity = self._calculate_text_similarity(main_text, related_text)
                consistency_scores.append(similarity)
            
            return sum(consistency_scores) / len(consistency_scores)
            
        except Exception as e:
            print(f"Error calculating consistency: {str(e)}")
            return 0.5
            
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        try:
            # Simple word overlap similarity
            words1 = set(text1.split())
            words2 = set(text2.split())
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            return len(intersection) / len(union) if union else 0.0
            
        except Exception as e:
            print(f"Error calculating similarity: {str(e)}")
            return 0.0
            
    def _is_left_leaning(self, article: Dict) -> bool:
        """Check if an article appears to be left-leaning"""
        text = article.get('metadata', {}).get('content', '').lower()
        left_terms = ['progressive', 'liberal', 'democrat', 'socialism', 'workers']
        return any(term in text for term in left_terms)
        
    def _is_right_leaning(self, article: Dict) -> bool:
        """Check if an article appears to be right-leaning"""
        text = article.get('metadata', {}).get('content', '').lower()
        right_terms = ['conservative', 'republican', 'traditional', 'freedom', 'patriot']
        return any(term in text for term in right_terms)
        
    def _get_historical_context(self, text: str) -> List[str]:
        """Get historical context for the article content"""
        # This would ideally query a knowledge base or timeline database
        return [
            "Similar topics have been debated in past policy discussions",
            "Historical trends show patterns in public discourse on this issue",
            "Previous related events provide context for current developments"
        ]
