#!/usr/bin/env python3
import asyncio
import asyncpg
import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

async def create_bucket(bucket_name, user_id):
    # Connect to the database using environment variables
    conn = await asyncpg.connect(
        user=os.environ.get("POSTGRES_USER", "postgres"),
        password=os.environ.get("POSTGRES_PASSWORD", "postgres"),
        database=os.environ.get("POSTGRES_DB", "memory_plugin"),
        host=os.environ.get("POSTGRES_HOST", "localhost")
    )
    
    # Create the bucket
    await conn.execute(
        "INSERT INTO buckets(name, user_id) VALUES($1, $2) ON CONFLICT (name, user_id) DO NOTHING",
        bucket_name, user_id
    )
    
    # Verify and get the bucket
    bucket = await conn.fetchrow("SELECT * FROM buckets WHERE name = $1 AND user_id = $2", bucket_name, user_id)
    
    # Close the connection
    await conn.close()
    
    return bucket

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python create_bucket.py <bucket_name> <user_id>")
        sys.exit(1)
    
    bucket_name = sys.argv[1]
    user_id = sys.argv[2]
    
    bucket = asyncio.run(create_bucket(bucket_name, user_id))
    
    if bucket:
        print(f"Bucket created: {bucket['name']} for user {bucket['user_id']}")
    else:
        print("Failed to create bucket")
