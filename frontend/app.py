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
            background: linear-gradient(90deg, #4CAF50, #2196F3);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 1em;
            text-align: center;
        }
        .subheader {
            color: #7c7c7c;
            font-size: 1.2em;
            margin-bottom: 2em;
            text-align: center;
        }
        .stTextInput > div > div > input {
            background-color: #2b2b2b;
            color: white;
            border-radius: 10px;
        }
        .stTextArea > div > div > textarea {
            background-color: #2b2b2b;
            color: white;
            border-radius: 10px;
        }
        .insight-box {
            background-color: rgba(76, 175, 80, 0.1);
            border-left: 3px solid #4CAF50;
            padding: 20px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .source-box {
            background-color: rgba(33, 150, 243, 0.1);
            border: 1px solid #2196F3;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
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
        
        # Add advanced settings
        with st.expander("⚙️ Advanced Settings"):
            st.checkbox("Enable real-time analysis", value=True)
            st.checkbox("Show source credibility scores", value=True)
            st.checkbox("Include historical context", value=True)
            st.selectbox(
                "Language model",
                ["Mistral-7B", "GPT-3.5", "Claude-2"],
                index=0
            )
    
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
            
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Update progress
            status_text.text("Finding related articles...")
            progress_bar.progress(25)
            
            # Search for related articles
            related_articles = components["searcher"].find_related(
                article_data,
                top_k=search_depth
            )
            
            status_text.text("Analyzing bias and sentiment...")
            progress_bar.progress(50)
            
            # Analyze bias
            bias_results = components["bias_detector"].analyze(
                article_data,
                related_articles
            )
            
            status_text.text("Generating insights...")
            progress_bar.progress(75)
            
            # Create tabs for different result sections
            tabs = st.tabs([
                "📊 Analysis Results",
                "🔍 Related Articles",
                "⚖️ Bias Meter",
                "📈 Insights"
            ])
            
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
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(
                                    f"""
                                    <div class="source-box">
                                        <p><strong>🌐 Source:</strong> {article.get('metadata', {}).get('domain', 'Unknown')}</p>
                                        <p><strong>📈 Relevance Score:</strong> {article.get('final_score', 0):.2f}</p>
                                        <p>{article.get('metadata', {}).get('content', '')[:500]}...</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                            with col2:
                                if article.get('metadata', {}).get('url'):
                                    st.markdown(f"🔗 [Read full article]({article['metadata']['url']})")
                                st.metric(
                                    "Source Credibility",
                                    f"{article.get('credibility_score', 0.7):.1%}"
                                )
                                
            with tabs[2]:
                if "bias_score" in bias_results:
                    components["bias_meter"].display_bias_meter(bias_results["bias_score"])
                    
            with tabs[3]:
                # Display key insights
                st.subheader("🎯 Key Insights")
                insights = bias_results.get('key_findings', [])
                for insight in insights:
                    st.markdown(
                        f"""
                        <div class="insight-box">
                            <p>✨ {insight}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                # Display fact-checking results
                if bias_results.get('fact_checks'):
                    st.subheader("✅ Fact Checking Results")
                    for fact in bias_results['fact_checks']:
                        cols = st.columns([4, 1])
                        with cols[0]:
                            st.info(f"🔍 {fact['claim']}")
                            st.markdown(f"💡 **Verdict:** {fact['verdict']}")
                        with cols[1]:
                            st.metric(
                                "Confidence",
                                f"{fact['confidence']:.1%}"
                            )
            
            # Clear progress indicators
            progress_bar.progress(100)
            status_text.empty()
            
            # Show export options
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button(
                    "📥 Download Report",
                    "Report data...",
                    "report.pdf"
                )
            with col2:
                st.button("📧 Share Results")
            with col3:
                st.button("💾 Save Analysis")

if __name__ == "__main__":
    main()
