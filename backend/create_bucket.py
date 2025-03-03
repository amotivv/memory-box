#!/usr/bin/env python3
"""
Memory Box - A semantic memory storage and retrieval system
Copyright (C) 2025 amotivv, inc.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
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
