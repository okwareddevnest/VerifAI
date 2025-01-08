import streamlit as st
import pandas as pd
from typing import Dict, List

class ResultsDisplay:
    def __init__(self):
        """Initialize the results display component"""
        pass
        
    def show_results(self, article_data: Dict, related_articles: List[Dict], bias_results: Dict):
        """Display the analysis results in the Streamlit UI"""
        st.header("Analysis Results")
        
        # Display error if present
        if "error" in bias_results:
            st.error(f"Analysis Error: {bias_results['error']}")
            return
            
        # Display bias score and political leaning
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Bias Score", f"{bias_results.get('bias_score', 0.5):.2f}")
        with col2:
            st.metric("Political Leaning", bias_results.get('political_leaning', 'Unknown'))
            
        # Display sentiment analysis
        st.subheader("Sentiment Analysis")
        sentiment_scores = bias_results.get('sentiment_scores', {})
        cols = st.columns(4)
        with cols[0]:
            st.metric("Positive", f"{sentiment_scores.get('pos', 0):.2f}")
        with cols[1]:
            st.metric("Neutral", f"{sentiment_scores.get('neu', 0):.2f}")
        with cols[2]:
            st.metric("Negative", f"{sentiment_scores.get('neg', 0):.2f}")
        with cols[3]:
            st.metric("Overall", bias_results.get('sentiment', 'neutral'))
            
        # Display bias indicators
        if bias_results.get('bias_indicators'):
            st.subheader("Bias Indicators")
            for indicator in bias_results['bias_indicators']:
                st.info(indicator)
                
        # Display related articles
        if related_articles:
            st.subheader("Related Articles")
            for article in related_articles:
                with st.expander(article.get('metadata', {}).get('title', 'Related Article')):
                    st.write(f"Source: {article.get('metadata', {}).get('domain', 'Unknown')}")
                    st.write(f"Similarity Score: {article.get('score', 0):.2f}")
                    st.write(article.get('metadata', {}).get('content', '')[:500] + "...")
                    if article.get('metadata', {}).get('url'):
                        st.markdown(f"[Read more]({article['metadata']['url']})")
