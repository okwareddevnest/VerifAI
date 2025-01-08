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
        
        # Set up required tables
        self.setup_tables()
    
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
            
            # Use warehouse if specified
            if os.getenv("SNOWFLAKE_WAREHOUSE"):
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
                political_leaning VARCHAR,
                sentiment VARCHAR,
                analysis_data VARIANT,
                created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
                FOREIGN KEY (article_id) REFERENCES NEWS_ARTICLES(id)
            )
            """)
            
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS ARTICLE_RELATIONSHIPS (
                source_id VARCHAR NOT NULL,
                related_id VARCHAR NOT NULL,
                similarity_score FLOAT,
                created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
                PRIMARY KEY (source_id, related_id),
                FOREIGN KEY (source_id) REFERENCES NEWS_ARTICLES(id),
                FOREIGN KEY (related_id) REFERENCES NEWS_ARTICLES(id)
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
            embedding = self.model.encode(text)
            
            # Convert numpy array to list of floats and then to string
            embedding_str = ','.join(map(str, embedding.tolist()))
            
            # Store embedding
            cursor.execute("""
            INSERT INTO ARTICLE_EMBEDDINGS (article_id, embedding)
            SELECT %s, ARRAY_CONSTRUCT_FROM_STRING(%s, ',')
            """, (
                article_id,
                embedding_str
            ))
            
            return article_id
            
        finally:
            cursor.close()
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Perform semantic search using embeddings"""
        cursor = self.conn.cursor()
        
        try:
            # Generate query embedding
            query_embedding = np.array(self.model.encode(query))
            
            # Fetch all articles and embeddings
            cursor.execute("""
            SELECT 
                a.id, a.title, a.content, a.url, a.domain,
                ARRAY_TO_STRING(e.embedding, ',') as embedding_str
            FROM NEWS_ARTICLES a
            JOIN ARTICLE_EMBEDDINGS e ON a.id = e.article_id
            """)
            
            results = []
            for row in cursor.fetchall():
                try:
                    # Convert embedding string to numpy array
                    embedding_values = [float(x) for x in row[5].split(',')]
                    article_embedding = np.array(embedding_values)
                    
                    # Ensure both embeddings have the same shape
                    if query_embedding.shape == article_embedding.shape:
                        # Calculate cosine similarity
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
                except (ValueError, TypeError, AttributeError) as e:
                    print(f"Error processing embedding: {str(e)}")
                    continue
            
            # Sort by similarity and return top_k
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]
            
        finally:
            cursor.close()
    
    def store_analysis(self, article_data: Dict, bias_results: Dict, related_articles: List[Dict]) -> str:
        """Store analysis results in Snowflake"""
        cursor = self.conn.cursor()
        
        try:
            # Generate unique ID for the analysis
            analysis_id = os.urandom(16).hex()
            
            # Store article
            cursor.execute("""
            INSERT INTO NEWS_ARTICLES (
                id, title, content, url, domain, created_at
            ) VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP())
            """, (
                analysis_id,
                article_data.get('title', ''),
                article_data.get('text', ''),
                article_data.get('url', ''),
                article_data.get('domain', '')
            ))
            
            # Store analysis results
            cursor.execute("""
            INSERT INTO ANALYSIS_RESULTS (
                article_id,
                bias_score,
                political_leaning,
                sentiment,
                analysis_data,
                created_at
            ) VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP())
            """, (
                analysis_id,
                bias_results.get('bias_score', 0.0),
                bias_results.get('political_leaning', 'unknown'),
                bias_results.get('sentiment', 'neutral'),
                json.dumps(bias_results)
            ))
            
            # Store related articles
            for article in related_articles:
                related_id = os.urandom(16).hex()
                metadata = article.get('metadata', {})
                
                cursor.execute("""
                INSERT INTO NEWS_ARTICLES (
                    id, title, content, url, domain, created_at
                ) VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP())
                """, (
                    related_id,
                    metadata.get('title', ''),
                    metadata.get('content', ''),
                    metadata.get('url', ''),
                    metadata.get('domain', '')
                ))
                
                # Store relationship
                cursor.execute("""
                INSERT INTO ARTICLE_RELATIONSHIPS (
                    source_id,
                    related_id,
                    similarity_score,
                    created_at
                ) VALUES (%s, %s, %s, CURRENT_TIMESTAMP())
                """, (
                    analysis_id,
                    related_id,
                    article.get('score', 0.0)
                ))
            
            # Commit the transaction
            self.conn.commit()
            return analysis_id
            
        except Exception as e:
            print(f"Error storing analysis: {str(e)}")
            self.conn.rollback()
            raise
        finally:
            cursor.close()
    
    def close(self):
        """Close Snowflake connection"""
        if self.conn:
            self.conn.close()
    
    def generate_analysis(self, article_text: str, related_articles: List[Dict], prompt_template: str = None) -> Dict:
        """Generate analysis using basic NLP techniques"""
        try:
            # Calculate basic metrics
            words = article_text.lower().split()
            total_words = len(words)
            
            # Define bias indicators
            bias_indicators = {
                'emotional': ['outrageous', 'shocking', 'terrible', 'amazing', 'incredible'],
                'partisan': ['leftist', 'rightist', 'liberal', 'conservative', 'radical'],
                'loaded': ['regime', 'elite', 'conspiracy', 'propaganda']
            }
            
            # Count bias indicators
            bias_counts = {category: sum(1 for word in words if word in terms) 
                         for category, terms in bias_indicators.items()}
            
            # Calculate bias score based on indicator presence
            total_bias_words = sum(bias_counts.values())
            bias_score = min(1.0, total_bias_words / (total_words * 0.1)) if total_words > 0 else 0.5
            
            # Determine political leaning based on word usage
            partisan_words = {
                'left': ['progressive', 'liberal', 'democrat', 'socialism'],
                'right': ['conservative', 'republican', 'traditional', 'freedom']
            }
            
            left_count = sum(1 for word in words if word in partisan_words['left'])
            right_count = sum(1 for word in words if word in partisan_words['right'])
            
            if abs(left_count - right_count) < 2:
                leaning = "center"
            elif left_count > right_count:
                leaning = "left" if left_count - right_count > 3 else "center-left"
            else:
                leaning = "right" if right_count - left_count > 3 else "center-right"
            
            # Collect bias indicators found
            found_indicators = []
            for category, count in bias_counts.items():
                if count > 0:
                    found_indicators.append(f"Found {count} {category} language patterns")
            
            # Compare with related articles
            source_comparison = "Multiple sources available" if len(related_articles) > 2 else "Limited sources"
            
            return {
                "bias_score": bias_score,
                "political_leaning": leaning,
                "bias_indicators": found_indicators,
                "sentiment": "neutral",  # This will be updated by VADER in BiasDetector
                "factual_accuracy": "Based on available sources",
                "source_comparison": source_comparison
            }
            
        except Exception as e:
            print(f"Error in analysis: {str(e)}")
            return {
                "error": str(e),
                "bias_score": 0.5,
                "political_leaning": "unknown",
                "sentiment": "neutral"
            } 