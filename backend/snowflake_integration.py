import snowflake.connector
from typing import Dict, List, Optional
import os
import json
from sentence_transformers import SentenceTransformer
import numpy as np
from dotenv import load_dotenv

class SnowflakeManager:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Validate required environment variables
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
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
            
        # Extract region from account identifier
        account = os.getenv("SNOWFLAKE_ACCOUNT")
        if "." in account:
            account_parts = account.split(".")
            account_id = account_parts[0]
            region = ".".join(account_parts[1:])
        else:
            account_id = account
            region = None
        
        # Initialize Snowflake connection
        print("Initializing Snowflake connection...")
        connection_params = {
            "account": account_id,
            "user": os.getenv("SNOWFLAKE_USER"),
            "password": os.getenv("SNOWFLAKE_PASSWORD"),
            "role": os.getenv("SNOWFLAKE_ROLE"),
        }
        
        if region:
            connection_params["region"] = region
            
        self.conn = snowflake.connector.connect(**connection_params)
        
        # Initialize sentence transformer for embeddings
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # After connection is established, set up database and schema
        self._ensure_database_exists()
    
    def _ensure_database_exists(self):
        """Ensure database and schema exist"""
        cursor = self.conn.cursor()
        try:
            # Create database if not exists
            print("Setting up database...")
            cursor.execute(f"""
            CREATE DATABASE IF NOT EXISTS {os.getenv("SNOWFLAKE_DATABASE")}
            """)
            
            # Use database
            cursor.execute(f"""
            USE DATABASE {os.getenv("SNOWFLAKE_DATABASE")}
            """)
            
            # Create schema if not exists
            cursor.execute("""
            CREATE SCHEMA IF NOT EXISTS ML
            """)
            
            # Use schema
            cursor.execute("""
            USE SCHEMA ML
            """)
            
            # Create and use warehouse if specified
            if os.getenv("SNOWFLAKE_WAREHOUSE"):
                cursor.execute(f"""
                CREATE WAREHOUSE IF NOT EXISTS {os.getenv("SNOWFLAKE_WAREHOUSE")}
                WITH WAREHOUSE_SIZE = 'XSMALL'
                AUTO_SUSPEND = 60
                AUTO_RESUME = TRUE
                """)
                
                cursor.execute(f"""
                USE WAREHOUSE {os.getenv("SNOWFLAKE_WAREHOUSE")}
                """)
            
            print("Database setup completed successfully")
            
        except Exception as e:
            print(f"Error during database setup: {str(e)}")
            raise
        finally:
            cursor.close()
    
    def setup_tables(self):
        """Set up required tables if they don't exist"""
        cursor = self.conn.cursor()
        
        try:
            print("Creating tables...")
            # Create tables if they don't exist
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS NEWS_ARTICLES (
                id VARCHAR NOT NULL,
                title VARCHAR,
                content TEXT,
                url VARCHAR,
                domain VARCHAR,
                published_date TIMESTAMP_NTZ,
                created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
                PRIMARY KEY (id)
            )
            """)
            
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS ARTICLE_EMBEDDINGS (
                article_id VARCHAR NOT NULL,
                embedding ARRAY,
                created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
                FOREIGN KEY (article_id) REFERENCES NEWS_ARTICLES(id)
            )
            """)
            
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS ANALYSIS_RESULTS (
                article_id VARCHAR NOT NULL,
                bias_score FLOAT,
                confidence FLOAT,
                sentiment VARCHAR,
                analysis_data VARIANT,
                created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
                FOREIGN KEY (article_id) REFERENCES NEWS_ARTICLES(id)
            )
            """)
            
            print("Tables created successfully")
            
        except Exception as e:
            print(f"Error creating tables: {str(e)}")
            raise
        finally:
            cursor.close()
    
    def index_article(self, article_data: Dict) -> str:
        """Index an article in Snowflake"""
        cursor = self.conn.cursor()
        
        try:
            # Generate article ID if not provided
            article_id = article_data.get('id', os.urandom(16).hex())
            
            # Insert article
            cursor.execute("""
            INSERT INTO NEWS_ARTICLES (id, title, content, url, domain)
            VALUES (%s, %s, %s, %s, %s)
            """, (
                article_id,
                article_data.get('title'),
                article_data.get('text'),
                article_data.get('url'),
                article_data.get('domain')
            ))
            
            # Generate and store embedding
            text = f"{article_data.get('title', '')} {article_data.get('text', '')}"
            embedding = self.model.encode(text).tolist()
            
            # Convert embedding to Snowflake array format
            cursor.execute("""
            INSERT INTO ARTICLE_EMBEDDINGS (article_id, embedding)
            SELECT %s, ARRAY_CONSTRUCT(%s)
            """, (
                article_id,
                ','.join(map(str, embedding))
            ))
            
            return article_id
            
        finally:
            cursor.close()
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Perform semantic search using embeddings"""
        cursor = self.conn.cursor()
        
        try:
            # Generate query embedding
            query_embedding = self.model.encode(query).tolist()
            
            # Fetch all articles and embeddings
            cursor.execute("""
            SELECT 
                a.id, a.title, a.content, a.url, a.domain,
                e.embedding
            FROM NEWS_ARTICLES a
            JOIN ARTICLE_EMBEDDINGS e ON a.id = e.article_id
            """)
            
            results = []
            for row in cursor.fetchall():
                article_embedding = np.array(row[5])
                similarity = np.dot(query_embedding, article_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(article_embedding)
                )
                
                results.append({
                    "score": float(similarity),
                    "metadata": {
                        "id": row[0],
                        "title": row[1],
                        "content": row[2],
                        "url": row[3],
                        "domain": row[4]
                    }
                })
            
            # Sort by similarity and return top_k
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]
            
        finally:
            cursor.close()
    
    def store_analysis(self, article_id: str, analysis_results: Dict):
        """Store analysis results"""
        cursor = self.conn.cursor()
        
        try:
            cursor.execute("""
            INSERT INTO ANALYSIS_RESULTS 
            (article_id, bias_score, confidence, sentiment, analysis_data)
            VALUES (%s, %s, %s, %s, %s)
            """, (
                article_id,
                analysis_results.get('bias_score', 0.0),
                analysis_results.get('confidence', 0.0),
                analysis_results.get('sentiment', 'neutral'),
                json.dumps(analysis_results)
            ))
            
        finally:
            cursor.close()
    
    def close(self):
        """Close Snowflake connection"""
        if self.conn:
            self.conn.close() 