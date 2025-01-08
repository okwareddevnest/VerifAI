import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import requests
from datetime import datetime, timedelta
from newsapi import NewsApiClient

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Load environment variables
load_dotenv(project_root / '.env')

from backend.snowflake_integration import SnowflakeManager
from backend.process import ArticleProcessor
import json
from typing import List, Dict
import time
from tqdm import tqdm

def load_sample_articles() -> List[Dict]:
    """Load real news articles from various sources"""
    newsapi = NewsApiClient(api_key=os.getenv('NEWS_API_KEY'))
    articles = []
    
    # Topics to fetch articles about
    topics = [
        "artificial intelligence ethics",
        "climate change policy",
        "global economics",
        "healthcare innovation",
        "technology impact",
        "political analysis",
        "scientific research",
        "social media influence"
    ]
    
    # Set date range (last 7 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    print("Fetching articles from NewsAPI...")
    for topic in tqdm(topics, desc="Topics"):
        try:
            response = newsapi.get_everything(
                q=topic,
                from_param=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d'),
                language='en',
                sort_by='relevancy',
                page_size=5  # 5 articles per topic
            )
            
            if response['status'] == 'ok':
                for article in response['articles']:
                    # Skip articles without content
                    if not article['content'] or not article['title']:
                        continue
                        
                    articles.append({
                        "url": article['url'],
                        "title": article['title'],
                        "text": f"{article['description']} {article['content']}",
                        "domain": article['source']['name'],
                        "published_date": article['publishedAt']
                    })
            
            # Rate limiting
            time.sleep(1)
            
        except Exception as e:
            print(f"Error fetching articles for topic '{topic}': {str(e)}")
            continue
    
    print(f"Fetched {len(articles)} articles from {len(topics)} topics")
    return articles

def main():
    print("Initializing Cortex Search setup...")
    
    # Verify environment variables
    required_vars = [
        "SNOWFLAKE_ACCOUNT",
        "SNOWFLAKE_USER",
        "SNOWFLAKE_PASSWORD",
        "SNOWFLAKE_WAREHOUSE",
        "SNOWFLAKE_DATABASE",
        "NEWS_API_KEY"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print("Error: Missing required environment variables:")
        for var in missing_vars:
            print(f"- {var}")
        sys.exit(1)
    
    # Initialize components
    try:
        snowflake = SnowflakeManager()
        processor = ArticleProcessor()
        
        # Set up tables
        print("Creating tables...")
        snowflake.setup_tables()
        
        # Load and process articles
        print("Loading articles from news sources...")
        articles = load_sample_articles()
        
        if not articles:
            print("Error: No articles were fetched. Check your NEWS_API_KEY and internet connection.")
            sys.exit(1)
        
        # Index articles
        print("Indexing articles...")
        for article in tqdm(articles, desc="Indexing"):
            try:
                # Process article
                processed_article = processor.process_text(article["text"])
                processed_article["url"] = article["url"]
                processed_article["title"] = article["title"]
                processed_article["domain"] = article["domain"]
                
                # Index in Snowflake
                doc_id = snowflake.index_article(processed_article)
                print(f"Indexed article: {article['title'][:60]}... [ID: {doc_id}]")
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error processing article '{article['title'][:30]}...': {str(e)}")
                continue
        
        print("\nSetup completed successfully!")
        print(f"Indexed {len(articles)} articles from various news sources")
        
    except Exception as e:
        print(f"Error during setup: {str(e)}")
        sys.exit(1)
    finally:
        if 'snowflake' in locals():
            snowflake.close()

if __name__ == "__main__":
    main() 