import streamlit as st
import pandas as pd

class ResultsDisplay:
    def __init__(self):
        self.sentiment_colors = {
            "positive": "🟢",
            "neutral": "⚪",
            "negative": "🔴"
        }
    
    def _display_summary(self, summary):
        st.subheader("Article Summary")
        st.write(summary)
    
    def _display_fact_checks(self, fact_checks):
        st.subheader("Fact Check Results")
        
        for fact in fact_checks:
            with st.expander(f"Claim: {fact['claim']}", expanded=True):
                st.markdown(f"""
                **Verdict:** {fact['verdict']}  
                **Confidence:** {fact['confidence']:.1%}  
                **Source:** {fact['source']}
                """)
                
                if fact.get('explanation'):
                    st.markdown("**Explanation:**")
                    st.write(fact['explanation'])
    
    def _display_related_articles(self, related_articles):
        st.subheader("Related Articles")
        
        # Create a dataframe for better display
        df = pd.DataFrame(related_articles)
        
        for idx, article in df.iterrows():
            with st.expander(f"{article['title']}", expanded=False):
                st.markdown(f"""
                **Source:** {article['source']}  
                **Date:** {article['date']}  
                **Agreement Score:** {article['agreement_score']:.1%}  
                **Sentiment:** {self.sentiment_colors[article['sentiment']]} {article['sentiment'].title()}
                
                {article['summary']}
                
                [Read Full Article]({article['url']})
                """)
    
    def _display_language_analysis(self, language_analysis):
        st.subheader("Language Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Emotional Language**")
            for emotion, score in language_analysis['emotional_scores'].items():
                st.progress(score, text=f"{emotion.title()}: {score:.1%}")
        
        with col2:
            st.markdown("**Writing Style**")
            for style, score in language_analysis['style_metrics'].items():
                st.progress(score, text=f"{style.title()}: {score:.1%}")
    
    def show_analysis(self, results):
        """Display the complete analysis results in a structured format"""
        
        # Main container for results
        with st.container():
            # Display article summary
            if "summary" in results:
                self._display_summary(results["summary"])
            
            # Display fact checking results
            if "fact_checks" in results:
                self._display_fact_checks(results["fact_checks"])
            
            # Display language analysis
            if "language_analysis" in results:
                self._display_language_analysis(results["language_analysis"])
            
            # Display related articles
            if "related_articles" in results:
                self._display_related_articles(results["related_articles"])
            
            # Display key takeaways
            if "key_takeaways" in results:
                st.subheader("Key Takeaways")
                for takeaway in results["key_takeaways"]:
                    st.markdown(f"- {takeaway}")
