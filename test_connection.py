import os
from dotenv import load_dotenv
import snowflake.connector

# Load environment variables
load_dotenv()

# Test connection
try:
    print("Connecting to Snowflake...")
    conn = snowflake.connector.connect(
        account=os.getenv('SNOWFLAKE_ACCOUNT'),
        user=os.getenv('SNOWFLAKE_USER'),
        password=os.getenv('SNOWFLAKE_PASSWORD'),
        role=os.getenv('SNOWFLAKE_ROLE'),
    )
    
    # Test query
    cursor = conn.cursor()
    cursor.execute('SELECT CURRENT_VERSION()')
    version = cursor.fetchone()[0]
    print(f"Successfully connected to Snowflake!")
    print(f"Snowflake version: {version}")
    
    cursor.close()
    conn.close()
    
except Exception as e:
    print(f"Connection failed: {str(e)}")
    print("\nDebug information:")
    print(f"Account: {os.getenv('SNOWFLAKE_ACCOUNT')}")
    print(f"User: {os.getenv('SNOWFLAKE_USER')}")
    print(f"Role: {os.getenv('SNOWFLAKE_ROLE')}") 