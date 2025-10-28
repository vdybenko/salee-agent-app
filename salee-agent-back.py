from google.cloud import bigquery
import pandas as pd
import os

# Set your project ID here
PROJECT_ID = "salee-chrome-extention"  # Replace with your actual project ID

try:
    # Initialize BigQuery client
    client = bigquery.Client(project=PROJECT_ID)
    print(f"Connected to BigQuery project: {PROJECT_ID}")
    
    # Example query - replace with your actual query
    query = """
    SELECT *
    FROM `salee-chrome-extention.SaleeAgent.conversations_embedded_extended`
    LIMIT 1000
    """
    
    # Execute query
    df = client.query(query).to_dataframe()
    print("Query executed successfully!")
    print(df.head())
    
except Exception as e:
    print(f"Error: {e}")
    print("\nTo fix this, you need to:")
    print("1. Set up Google Cloud credentials:")
    print("   gcloud auth application-default login")
    print("2. Or set the GOOGLE_APPLICATION_CREDENTIALS environment variable")
    print("3. Or replace 'your-project-id' with your actual Google Cloud project ID")