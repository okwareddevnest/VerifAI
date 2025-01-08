import streamlit as st
from dotenv import load_dotenv
import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Load environment variables at startup
load_dotenv()

# Validate Snowflake configuration
required_vars = [
    "SNOWFLAKE_ACCOUNT",
    "SNOWFLAKE_USER",
    "SNOWFLAKE_PASSWORD",
    "SNOWFLAKE_ROLE",
    "SNOWFLAKE_DATABASE",
    "SNOWFLAKE_WAREHOUSE"
]

missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    st.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    st.stop()

# Import after environment validation and path setup
from backend.search import NewsSearcher
from backend.bias_analysis import BiasDetector
from frontend.components.bias_meter import BiasAnalyzer
from frontend.components.results_display import ResultsDisplay

def main():
    st.set_page_config(
        page_title="VerifAI: Media Bias & Fake News Detector",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("VerifAI: Media Bias & Fake News Detector")
    st.markdown("""
    Verify news articles by cross-referencing multiple sources and analyzing potential bias in language.
    Enter a URL or paste article text below to get started.
    """)
    
    # Initialize components
    searcher = NewsSearcher()
    bias_detector = BiasDetector()
    bias_meter = BiasAnalyzer()
    results_display = ResultsDisplay()
    
    # Sidebar settings
    with st.sidebar:
        st.header("Analysis Settings")
        search_depth = st.slider("Search Depth", min_value=3, max_value=10, value=5,
                               help="Number of related articles to analyze")
        fact_check_confidence = st.slider("Fact-Check Confidence", min_value=0.0, max_value=1.0, value=0.7,
                                        help="Minimum confidence threshold for fact-checking")
    
    # Input section
    input_type = st.radio("Input Type", ["Article URL", "Article Text"])
    
    if input_type == "Article URL":
        article_url = st.text_input("Enter article URL")
        if article_url:
            with st.spinner("Analyzing article..."):
                # Process URL
                article_data = searcher.process_url(article_url)
                if article_data:
                    # Find related articles
                    related_articles = searcher.find_related(article_data, top_k=search_depth)
                    # Analyze bias
                    bias_results = bias_detector.analyze(article_data, related_articles)
                    # Display results
                    results_display.show_results(article_data, related_articles, bias_results)
                    bias_meter.display_bias_meter(bias_results["bias_score"])
                else:
                    st.error("Could not process the article URL. Please check the URL and try again.")
    else:
        article_text = st.text_area("Enter article text")
        if article_text:
            with st.spinner("Analyzing text..."):
                # Process text
                article_data = {"text": article_text}
                # Find related articles
                related_articles = searcher.find_related(article_data, top_k=search_depth)
                # Analyze bias
                bias_results = bias_detector.analyze(article_data, related_articles)
                # Display results
                results_display.show_results(article_data, related_articles, bias_results)
                bias_meter.display_bias_meter(bias_results["bias_score"])

if __name__ == "__main__":
    main()
