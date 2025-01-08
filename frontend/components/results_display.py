import streamlit as st
import pandas as pd
from typing import Dict, List

class ResultsDisplay:
    def __init__(self):
        """Initialize the results display component"""
        pass
        
    def show_results(self, article_data: Dict, related_articles: List[Dict], bias_results: Dict):
        """Display the analysis results in the Streamlit UI"""
        st.header("� Analysis Results")
        
        # Display error if present
        if "error" in bias_results:
            st.error(f"Analysis Error: {bias_results['error']}")
            return
            
        # Create columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "🎯 Bias Score",
                f"{bias_results.get('bias_score', 0.5):.2f}",
                help="Score from -1 (far left) to 1 (far right)"
            )
        with col2:
            st.metric(
                "🌐 Political Leaning",
                bias_results.get('political_leaning', 'Unknown').replace('_', ' ').title(),
                help="Overall political orientation of the content"
            )
        with col3:
            st.metric(
                "💭 Overall Sentiment",
                bias_results.get('sentiment', 'neutral').title(),
                help="General emotional tone of the content"
            )
            
        # Display sentiment analysis details
        st.subheader("🔍 Sentiment Analysis")
        sentiment_scores = bias_results.get('sentiment_scores', {})
        cols = st.columns(4)
        with cols[0]:
            st.metric("😊 Positive", f"{sentiment_scores.get('pos', 0):.2f}")
        with cols[1]:
            st.metric("😐 Neutral", f"{sentiment_scores.get('neu', 0):.2f}")
        with cols[2]:
            st.metric("☹️ Negative", f"{sentiment_scores.get('neg', 0):.2f}")
        with cols[3]:
            st.metric("📊 Compound", f"{sentiment_scores.get('compound', 0):.2f}")
            
        # Display bias indicators
        if bias_results.get('bias_indicators'):
            st.subheader("⚠️ Bias Indicators")
            for indicator in bias_results['bias_indicators']:
                st.info(f"📌 {indicator}")
                
        # Display key findings if available
        if bias_results.get('key_findings'):
            st.subheader("🔑 Key Findings")
            for finding in bias_results['key_findings']:
                st.success(f"✨ {finding}")
                
        # Display recommendations if available
        if bias_results.get('recommendations'):
            st.subheader("💡 Recommendations")
            for recommendation in bias_results['recommendations']:
                st.warning(f"📝 {recommendation}")
                
        # Display source analysis if available
        if bias_results.get('source_analysis'):
            st.subheader("📰 Source Analysis")
            source_analysis = bias_results['source_analysis']
            cols = st.columns(2)
            with cols[0]:
                st.metric(
                    "🏢 Source Credibility",
                    source_analysis.get('credibility', 'Unknown'),
                    help="Reliability rating of the source"
                )
            with cols[1]:
                st.metric(
                    "✔️ Fact Consistency",
                    source_analysis.get('fact_consistency', 'Unknown'),
                    help="How well facts align with other sources"
                )
                
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
