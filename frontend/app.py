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

@st.cache_resource
def initialize_components():
    """Initialize components with caching to prevent multiple initializations"""
    return {
        "searcher": NewsSearcher(),
        "bias_detector": BiasDetector(),
        "bias_meter": BiasAnalyzer(),
        "results_display": ResultsDisplay()
    }

def main():
    st.set_page_config(
        page_title="VerifAI: Media Bias & Fake News Detector",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main-header {
            color: #4CAF50;
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 1em;
        }
        .subheader {
            color: #7c7c7c;
            font-size: 1.2em;
            margin-bottom: 2em;
        }
        .stTextInput > div > div > input {
            background-color: #2b2b2b;
            color: white;
        }
        .stTextArea > div > div > textarea {
            background-color: #2b2b2b;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Sidebar settings
    with st.sidebar:
        st.header("🛠️ Analysis Settings")
        st.markdown("---")
        
        search_depth = st.slider(
            "🔍 Search Depth",
            min_value=3,
            max_value=10,
            value=5,
            help="Number of related articles to analyze"
        )
        
        fact_check_confidence = st.slider(
            "✅ Fact-Check Confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            help="Minimum confidence threshold for fact-checking"
        )
        
        st.markdown("---")
        st.markdown("### 🎯 How it works")
        st.markdown("""
        1. 📝 Paste your article text
        2. 🤖 AI analyzes the content
        3. 🔄 Cross-references sources
        4. 📊 Generates bias report
        """)
    
    # Main content
    st.markdown('<h1 class="main-header">🔍 VerifAI: Media Bias & Fake News Detector</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subheader">Verify news articles by cross-referencing multiple sources and analyzing potential bias in language. '
        'Enter your article text below to get started.</p>',
        unsafe_allow_html=True
    )
    
    # Initialize components
    components = initialize_components()
    
    # Article text input with placeholder
    text = st.text_area(
        "📝 Enter article text",
        height=200,
        placeholder="Paste your article text here...",
        help="Paste the full text of the article you want to analyze"
    )
    
    if text:
        with st.spinner("🔄 Analyzing article..."):
            article_data = {"text": text, "title": text[:50] + "..."}
            
            # Search for related articles
            related_articles = components["searcher"].find_related(
                article_data,
                top_k=search_depth
            )
            
            # Analyze bias
            bias_results = components["bias_detector"].analyze(
                article_data,
                related_articles
            )
            
            # Create tabs for different result sections
            tabs = st.tabs(["📊 Analysis Results", "🔍 Related Articles", "⚖️ Bias Meter"])
            
            with tabs[0]:
                components["results_display"].show_results(
                    article_data,
                    related_articles,
                    bias_results
                )
                
            with tabs[1]:
                if related_articles:
                    for article in related_articles:
                        with st.expander(f"📰 {article.get('metadata', {}).get('title', 'Related Article')}"):
                            st.write(f"🌐 Source: {article.get('metadata', {}).get('domain', 'Unknown')}")
                            st.write(f"📈 Similarity Score: {article.get('score', 0):.2f}")
                            st.write(article.get('metadata', {}).get('content', '')[:500] + "...")
                            if article.get('metadata', {}).get('url'):
                                st.markdown(f"🔗 [Read full article]({article['metadata']['url']})")
                                
            with tabs[2]:
                if "bias_score" in bias_results:
                    components["bias_meter"].display_bias_meter(bias_results["bias_score"])

if __name__ == "__main__":
    main()
