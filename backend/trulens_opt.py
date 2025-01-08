from trulens_eval import Feedback, TruLlama
from trulens_eval.feedback import Groundedness
from trulens_eval.feedback.provider.openai import OpenAI
import numpy as np
from typing import Dict, List, Optional
import os

class TruLensOptimizer:
    def __init__(self):
        # Initialize TruLens components
        self.openai = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.groundedness = Groundedness(groundedness_provider=self.openai)
        
        # Define feedback functions
        self.feedback_fns = [
            Feedback(
                name="groundedness",
                fn=self.groundedness.groundedness_measure_with_cot_reasons,
                higher_is_better=True
            ),
            Feedback(
                name="relevance",
                fn=self.evaluate_relevance,
                higher_is_better=True
            ),
            Feedback(
                name="bias_detection",
                fn=self.evaluate_bias_detection,
                higher_is_better=True
            )
        ]
    
    def evaluate_relevance(self, 
                          article_data: Dict,
                          related_articles: List[Dict]) -> float:
        """Evaluate the relevance of retrieved articles"""
        if not related_articles:
            return 0.0
        
        relevance_scores = [
            article.get('relevance_score', 0.0)
            for article in related_articles
        ]
        
        return np.mean(relevance_scores)
    
    def evaluate_bias_detection(self,
                              bias_results: Dict,
                              ground_truth: Optional[Dict] = None) -> float:
        """Evaluate the accuracy of bias detection"""
        if ground_truth is None:
            # If no ground truth is provided, use confidence as score
            return bias_results.get('confidence', 0.0)
        
        # Calculate score based on ground truth
        predicted_score = bias_results.get('bias_score', 0.0)
        true_score = ground_truth.get('bias_score', 0.0)
        
        # Simple error calculation
        error = abs(predicted_score - true_score)
        score = 1.0 - min(error, 1.0)  # Normalize to [0,1]
        
        return score
    
    def evaluate_system(self,
                       article_data: Dict,
                       analysis_results: Dict,
                       related_articles: List[Dict],
                       ground_truth: Optional[Dict] = None) -> Dict:
        """Evaluate the overall system performance"""
        
        # Calculate individual scores
        groundedness_score = self.groundedness.groundedness_measure_with_cot_reasons(
            record=article_data,
            response=analysis_results
        )
        
        relevance_score = self.evaluate_relevance(
            article_data,
            related_articles
        )
        
        bias_detection_score = self.evaluate_bias_detection(
            analysis_results,
            ground_truth
        )
        
        # Combine scores
        overall_score = np.mean([
            groundedness_score,
            relevance_score,
            bias_detection_score
        ])
        
        return {
            'overall_score': overall_score,
            'groundedness': groundedness_score,
            'relevance': relevance_score,
            'bias_detection': bias_detection_score,
            'details': {
                'num_related_articles': len(related_articles),
                'confidence': analysis_results.get('confidence', 0.0)
            }
        }
    
    def optimize_parameters(self,
                          evaluation_results: List[Dict]) -> Dict:
        """Optimize system parameters based on evaluation results"""
        if not evaluation_results:
            return {}
        
        # Calculate optimal parameters
        # This is a placeholder - in a real implementation,
        # you'd want to use more sophisticated optimization techniques
        
        scores = np.array([r['overall_score'] for r in evaluation_results])
        best_idx = np.argmax(scores)
        best_result = evaluation_results[best_idx]
        
        # Extract parameters from best result
        optimal_params = {
            'relevance_threshold': 0.1,  # These would be actual parameters
            'bias_threshold': 0.7,       # from your best performing run
            'confidence_threshold': 0.8
        }
        
        return {
            'optimal_parameters': optimal_params,
            'best_score': best_result['overall_score'],
            'improvement': float(best_result['overall_score'] - np.mean(scores))
        }
