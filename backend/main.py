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

from fastapi import FastAPI, HTTPException, Depends, Header, Query, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union
import os
import httpx
import time
import asyncpg
import logging
import json
import sys
import asyncio
from datetime import datetime

from usage_tracking import UsageTrackingMiddleware, RequestBodyCaptureMiddleware, PoolAccessMiddleware

# Custom JSON formatter for Google Cloud compatibility
class GoogleCloudFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "severity": record.levelname,
            "message": super().format(record),
            "logger": record.name,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if available
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
            
        # Add any extra fields
        if hasattr(record, 'extra'):
            log_entry.update(record.extra)
            
        return json.dumps(log_entry)

# Configure root logger
def setup_logging(level=logging.INFO):
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    
    # Determine format based on environment
    if os.environ.get("APP_ENV") == "development":
        # Human-readable format for development
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    else:
        # JSON format for production/Google Cloud
        formatter = GoogleCloudFormatter()
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

# Create logger instance at module level
logger = setup_logging(
    level=logging.DEBUG if os.environ.get("APP_ENV") == "development" else logging.INFO
)

app = FastAPI(
    title="Memory Box API",
    description="A semantic memory storage and retrieval system",
    version="1.1.0",
    docs_url="/docs"
)

# Security scheme
security = HTTPBearer(auto_error=False)

# API configuration
API_PORT = int(os.environ.get("API_PORT", 8000))
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")
DEFAULT_BUCKET_NAME = os.environ.get("DEFAULT_BUCKET_NAME", "General")
SEARCH_RESULTS_LIMIT = int(os.environ.get("SEARCH_RESULTS_LIMIT", 10))
INITIAL_SEARCH_LIMIT = int(os.environ.get("INITIAL_SEARCH_LIMIT", 15))
KEYWORD_BOOST_FACTOR = float(os.environ.get("KEYWORD_BOOST_FACTOR", "0.05"))
EMBEDDING_TIMEOUT = float(os.environ.get("EMBEDDING_TIMEOUT", "30.0"))

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pool variable as global
pool = None

# Register our usage tracking middlewares in the correct order
app.add_middleware(UsageTrackingMiddleware)
app.add_middleware(PoolAccessMiddleware)
app.add_middleware(RequestBodyCaptureMiddleware)

# Database connection parameters
DATABASE_URL = f"postgresql://{os.environ.get('POSTGRES_USER')}:{os.environ.get('POSTGRES_PASSWORD')}@{os.environ.get('POSTGRES_HOST')}/{os.environ.get('POSTGRES_DB')}"

# Ollama embedding server URL
OLLAMA_URL = os.environ.get("OLLAMA_URL", "your_ollama_api_url")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mxbai-embed-large")

# Configure similarity threshold (lowered for better recall)
SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", "0.55"))

# Models
class Memory(BaseModel):
    text: str
    bucketId: Optional[str] = None

class MemoryResponse(BaseModel):
    id: int
    text: str
    bucket_id: str
    created_at: str
    similarity: Optional[float] = None

class MemoriesListResponse(BaseModel):
    """Response model for lists of memories to ensure consistent array serialization."""
    items: List[Dict[str, Any]]

class DebugInfo(BaseModel):
    query: str
    query_terms: List[str]
    threshold: float
    all_results: List[Dict[str, Any]]

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    debug: Optional[DebugInfo] = None

class BucketsListResponse(BaseModel):
    """Response model for lists of buckets."""
    items: List[Dict[str, Any]]

# Token verification helper
async def verify_token_and_bucket(token: str, bucket_id: str, conn) -> bool:
    """
    Verify if a token (user_id) and bucket combination is valid.
    
    Args:
        token: The user_id token
        bucket_id: The bucket name
        conn: Database connection
        
    Returns:
        True if valid, False otherwise
    """
    if not token or not bucket_id:
        return False
        
    # Check if the token and bucket combination exists
    bucket = await conn.fetchrow(
        "SELECT id FROM buckets WHERE name = $1 AND user_id = $2", 
        bucket_id, token
    )
    
    return bucket is not None

# Database connection pool
async def get_db_pool():
    return await asyncpg.create_pool(DATABASE_URL)

# Custom exception handler for better error responses
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    error_response = {
        "error": True,
        "status": exc.status_code,
        "detail": exc.detail
    }
    
    # Add more context for authentication errors
    if exc.status_code == 401:
        error_response["auth_required"] = True
        error_response["message"] = "Authentication required. Please provide a valid token."
    elif exc.status_code == 403:
        error_response["message"] = "You don't have permission to access this resource."
    
    # Log the error
    logger.warning(
        f"HTTP error: {exc.status_code}", 
        extra={
            "path": request.url.path,
            "detail": exc.detail,
            "client_ip": request.client.host if hasattr(request, 'client') else "unknown"
        }
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response
    )

@app.on_event("startup")
async def startup():
    logger.info("Application starting up")
    global pool
    try:
        pool = await get_db_pool()
        logger.info("Database connection pool established")
    except Exception as e:
        logger.error(f"Failed to establish database connection: {str(e)}", exc_info=True)
        raise

@app.on_event("shutdown")
async def shutdown():
    logger.info("Application shutting down")
    if pool:
        await pool.close()
        logger.info("Database connection pool closed")

# Helper to get embeddings from Ollama
async def get_embedding(text):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={
                "model": OLLAMA_MODEL,
                "prompt": text
            },
            timeout=EMBEDDING_TIMEOUT
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Embedding service error: {response.text}")
            
        result = response.json()
        return result["embedding"]

# Helper to extract key terms from a query
def extract_query_terms(query: str) -> set:
    # Remove common words that wouldn't be helpful for matching
    stopwords = {"the", "and", "or", "a", "an", "in", "on", "at", "for", "to", "of", "with", "by"}
    return {word.lower() for word in query.split() if word.lower() not in stopwords and len(word) > 2}

# Authentication helper for the docs and API
async def get_user_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials is None:
        raise HTTPException(status_code=401, detail="Missing token")
    token = credentials.credentials
    if not token:
        raise HTTPException(status_code=401, detail="Missing token")
    return token

# API endpoints
@app.post("/api/v2/memory")
async def add_memory(
    memory: Memory, 
    token: str = Depends(get_user_token)
):
    logger.info("Adding new memory", extra={"bucket_id": memory.bucketId})
    
    # Extract user_id from token (token is the user_id)
    user_id = token
    logger.debug("Processing memory for user", extra={"user_id": user_id})
    
    # Format text with date if needed
    today = datetime.now()
    if not memory.text.strip().startswith(str(today.year)):
        formatted_date = today.strftime("%Y-%m-%d")
        memory.text = f"{formatted_date}\n\n{memory.text}"
    
    # Get embedding for the memory
    embedding = await get_embedding(memory.text)
    
    # Format embedding for pgvector - convert Python list to proper pgvector format
    embedding_str = f"[{','.join(str(x) for x in embedding)}]"
    
    # Use default bucket if none provided
    bucket_id = memory.bucketId or DEFAULT_BUCKET_NAME
    
    async with pool.acquire() as conn:
        # Verify token and bucket combination
        bucket = await conn.fetchrow(
            "SELECT * FROM buckets WHERE name = $1 AND user_id = $2", 
            bucket_id, user_id
        )
        
        if not bucket:
            # If bucket doesn't exist, check if user has any buckets
            user_buckets = await conn.fetch(
                "SELECT name FROM buckets WHERE user_id = $1", 
                user_id
            )
            
            if not user_buckets:
                logger.warning("Unauthorized memory add attempt - invalid token", 
                              extra={"token_prefix": token[:4] + "..." if len(token) > 4 else token})
                raise HTTPException(status_code=401, detail="Invalid token")
            
            # Create the bucket for this user
            logger.info(f"Creating new bucket '{bucket_id}' for user", extra={"user_id": user_id})
            await conn.execute(
                "INSERT INTO buckets(name, user_id) VALUES($1, $2)",
                bucket_id, user_id
            )
        
        # Insert memory with properly formatted embedding
        memory_id = await conn.fetchval(
            """
            INSERT INTO memories(text, embedding, bucket_id, user_id)
            VALUES($1, $2::vector, $3, $4) RETURNING id
            """,
            memory.text, embedding_str, bucket_id, user_id
        )
    
    return {"id": memory_id, "text": memory.text}

@app.get("/api/v2/memory", response_model=Union[MemoriesListResponse, SearchResponse])
async def get_memories(
    query: Optional[str] = None, 
    all: Optional[bool] = False,
    bucketId: Optional[str] = None,
    token: str = Depends(get_user_token),
    debug: bool = Query(False, description="Return debug information about the search")
):
    # Log the request type
    if query:
        logger.info("Searching memories", extra={"query": query})
    elif all:
        logger.info("Fetching all memories")
    elif bucketId:
        logger.info("Fetching memories from bucket", extra={"bucket_id": bucketId})
    else:
        logger.info("Memory request with no parameters")
    
    # Extract user_id from token (token is the user_id)
    user_id = token
    logger.debug("Processing memory request for user", extra={"user_id": user_id})
    
    async with pool.acquire() as conn:
        # Verify token is valid by checking if user has any buckets
        user_buckets = await conn.fetch(
            "SELECT name FROM buckets WHERE user_id = $1", 
            user_id
        )
        
        if not user_buckets:
            logger.warning("Unauthorized memory retrieval - invalid token", 
                          extra={"token_prefix": token[:4] + "..." if len(token) > 4 else token})
            raise HTTPException(status_code=401, detail="Invalid token")
        
        # For bucket-specific operations, verify the bucket belongs to the user
        if bucketId:
            is_valid = await verify_token_and_bucket(token, bucketId, conn)
            
            if not is_valid:
                logger.warning("Unauthorized bucket access attempt", 
                              extra={"bucket": bucketId, "token_prefix": token[:4] + "..." if len(token) > 4 else token})
                raise HTTPException(status_code=403, detail="Access denied to this bucket")
        # If getting all memories
        if all:
            memories = await conn.fetch(
                "SELECT id, text, bucket_id, created_at FROM memories WHERE user_id = $1",
                user_id
            )
            memory_list = [dict(m) for m in memories]
            logger.debug(f"Returning {len(memory_list)} memories from all buckets")
            return MemoriesListResponse(items=memory_list)
        
        # If getting memories from a specific bucket
        if bucketId:
            memories = await conn.fetch(
                "SELECT id, text, bucket_id, created_at FROM memories WHERE bucket_id = $1 AND user_id = $2",
                bucketId, user_id
            )
            memory_list = [dict(m) for m in memories]
            logger.debug(f"Returning {len(memory_list)} memories from bucket {bucketId}")
            return MemoriesListResponse(items=memory_list)
        
        # If searching for memories
        if query:
            # Extract query terms for boosting
            query_terms = extract_query_terms(query)
            
            # Get embedding for the query
            query_embedding = await get_embedding(query)
            
            # Format query embedding for pgvector
            query_embedding_str = f"[{','.join(str(x) for x in query_embedding)}]"
            
            # Perform vector search, getting more results than needed for reranking
            memories = await conn.fetch(
                f"""
                SELECT id, text, bucket_id, created_at, 
                       1 - (embedding <=> $1::vector) AS similarity
                FROM memories
                WHERE user_id = $2 AND 1 - (embedding <=> $1::vector) > {SIMILARITY_THRESHOLD}
                ORDER BY similarity DESC
                LIMIT {INITIAL_SEARCH_LIMIT}
                """,
                query_embedding_str, user_id
            )
            
            # Convert to list of dictionaries for easier manipulation
            memories_list = [dict(m) for m in memories]
            
            # If no results using vector search, try fallback to text search
            if not memories_list:
                text_search_results = await conn.fetch(
                    """
                    SELECT id, text, bucket_id, created_at
                    FROM memories 
                    WHERE user_id = $1 AND text ILIKE $2
                    LIMIT 5
                    """,
                    user_id, f"%{query}%"
                )
                memories_list = [dict(m) for m in text_search_results]
                # Add a fixed similarity score for text search results
                for m in memories_list:
                    m["similarity"] = 0.5
            else:
                # Apply keyword boosting for vector search results
                for memory in memories_list:
                    text_lower = memory['text'].lower()
                    # Count how many query terms appear in the text
                    matches = sum(1 for term in query_terms if term in text_lower)
                    # Boost similarity score for exact matches
                    memory['similarity'] = min(1.0, float(memory['similarity']) + (matches * KEYWORD_BOOST_FACTOR))
            
            # Re-sort by adjusted similarity
            memories_list.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Limit to top results
            result_memories = memories_list[:SEARCH_RESULTS_LIMIT]
            
            # Return debug information if requested
            if debug:
                return SearchResponse(
                    results=result_memories,
                    debug=DebugInfo(
                        query=query,
                        query_terms=list(query_terms),
                        threshold=SIMILARITY_THRESHOLD,
                        all_results=[
                            {
                                "id": m["id"], 
                                "text": m["text"][:100] + "..." if len(m["text"]) > 100 else m["text"], 
                                "similarity": m["similarity"],
                                "bucket_id": m["bucket_id"]
                            } 
                            for m in memories_list
                        ]
                    )
                )
            
            # Return search results using the consistent response model
            logger.debug(f"Returning {len(result_memories)} search results")
            return MemoriesListResponse(items=result_memories)
    
    # If no parameters specified, return an empty list
    logger.debug("No query parameters specified, returning empty list")
    return MemoriesListResponse(items=[])

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Endpoint to list all buckets for a user
@app.get("/api/v2/buckets", response_model=BucketsListResponse)
async def get_buckets(token: str = Depends(get_user_token)):
    # Extract user_id from token (token is the user_id)
    user_id = token
    logger.debug("Retrieving buckets for user", extra={"user_id": user_id})
    
    async with pool.acquire() as conn:
        # Query that joins buckets with latest memory timestamps
        buckets_with_latest = await conn.fetch("""
            SELECT b.id, b.name, b.created_at, 
                   (SELECT MAX(created_at) 
                    FROM memories m 
                    WHERE m.bucket_id = b.name AND m.user_id = b.user_id) AS latest_memory_timestamp
            FROM buckets b
            WHERE b.user_id = $1
            ORDER BY latest_memory_timestamp DESC NULLS LAST
        """, user_id)
        
        # Convert to list of dictionaries
        bucket_list = [dict(b) for b in buckets_with_latest]
        
    logger.debug(f"Returning {len(bucket_list)} buckets")
    return BucketsListResponse(items=bucket_list)

# User-accessible usage stats endpoint
@app.get("/api/v2/usage")
async def user_usage_stats(token: str = Depends(get_user_token)):
    # User ID is the token
    user_id = token
    
    async with pool.acquire() as conn:
        # Check if tracking tables exist
        tables_exist = await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'usage_metrics'
            );
            """
        )
        
        if not tables_exist:
            return {"status": "usage tracking not configured"}
        
        # Get user's plan information
        user_plan = await conn.fetchrow(
            """
            SELECT up.user_id, up.plan_name, p.store_memory_limit, p.search_memory_limit,
                   p.embedding_limit, p.api_call_limit, p.storage_limit_bytes
            FROM user_plans up
            JOIN plans p ON up.plan_name = p.name
            WHERE up.user_id = $1
            """,
            user_id
        )
        
        if not user_plan:
            return {"status": "no plan assigned"}
        
        # Get current month's usage
        now = datetime.now()
        monthly_usage = await conn.fetchrow(
            """
            SELECT store_memory_count, search_memory_count, embedding_count, api_call_count, 
                   total_bytes_processed
            FROM monthly_usage
            WHERE user_id = $1 AND year = $2 AND month = $3
            """,
            user_id, now.year, now.month
        )
        
        # Get operation breakdown
        user_operations = await conn.fetch(
            """
            SELECT operation_type, COUNT(*) as count
            FROM usage_metrics
            WHERE user_id = $1
            GROUP BY operation_type
            ORDER BY count DESC
            """,
            user_id
        )
        
        # Check if this is a legacy user (for UI customization)
        is_legacy = user_plan["plan_name"] == "legacy" if user_plan else False
        
        return {
            "status": "active",
            "plan": user_plan["plan_name"] if user_plan else None,
            "is_legacy_user": is_legacy,
            "current_month_usage": dict(monthly_usage) if monthly_usage else {
                "store_memory_count": 0,
                "search_memory_count": 0,
                "embedding_count": 0,
                "api_call_count": 0,
                "total_bytes_processed": 0
            },
            "limits": {
                "store_memory_limit": user_plan["store_memory_limit"] if user_plan else None,
                "search_memory_limit": user_plan["search_memory_limit"] if user_plan else None,
                "embedding_limit": user_plan["embedding_limit"] if user_plan else None,
                "api_call_limit": user_plan["api_call_limit"] if user_plan else None,
                "storage_limit_bytes": user_plan["storage_limit_bytes"] if user_plan else None,
            } if not is_legacy else {"enforced": False, "message": "Legacy users have no enforced limits"},
            "operations_breakdown": [{"operation": row["operation_type"], "count": row["count"]} for row in user_operations]
        }

# Admin-only system stats endpoint
@app.get("/admin/system-stats")
async def admin_system_stats(authorization: str = Header(None)):
    # Check admin token
    token = authorization.replace("Bearer ", "") if authorization else None
    admin_token = os.environ.get("ADMIN_TOKEN")
    if not token or not admin_token or token != admin_token:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    async with pool.acquire() as conn:
        # Check if tables exist
        tables_exist = await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'usage_metrics'
            );
            """
        )
        
        if not tables_exist:
            return {"status": "usage tracking not configured"}
        
        # System-wide statistics
        total_metrics = await conn.fetchval("SELECT COUNT(*) FROM usage_metrics")
        user_count = await conn.fetchval("SELECT COUNT(*) FROM user_plans")
        plan_breakdown = await conn.fetch(
            """
            SELECT plan_name, COUNT(*) as user_count
            FROM user_plans
            GROUP BY plan_name
            """
        )
        
        # Operation stats
        top_operations = await conn.fetch(
            """
            SELECT operation_type, COUNT(*) as count
            FROM usage_metrics
            GROUP BY operation_type
            ORDER BY count DESC
            """
        )
        
        # Most active users
        active_users = await conn.fetch(
            """
            SELECT user_id, COUNT(*) as operation_count
            FROM usage_metrics
            GROUP BY user_id
            ORDER BY operation_count DESC
            LIMIT 10
            """
        )
        
        return {
            "status": "active",
            "total_operations_tracked": total_metrics,
            "total_users": user_count,
            "plan_distribution": [{"plan": row["plan_name"], "users": row["user_count"]} for row in plan_breakdown],
            "operation_breakdown": [{"operation": row["operation_type"], "count": row["count"]} for row in top_operations],
            "most_active_users": [{"user_id": row["user_id"], "operations": row["operation_count"]} for row in active_users]
        }

# Admin endpoint to update a user's plan
@app.put("/admin/user-plans/{user_id}")
async def update_user_plan(
    user_id: str, 
    plan_name: str = Query(..., description="New plan name"),
    authorization: str = Header(None)
):
    # Check admin token
    token = authorization.replace("Bearer ", "") if authorization else None
    admin_token = os.environ.get("ADMIN_TOKEN")
    if not token or not admin_token or token != admin_token:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    async with pool.acquire() as conn:
        # Validate plan exists
        plan_exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM plans WHERE name = $1)",
            plan_name
        )
        
        if not plan_exists:
            raise HTTPException(status_code=400, detail=f"Plan '{plan_name}' does not exist")
        
        # Update user plan
        await conn.execute(
            """
            INSERT INTO user_plans (user_id, plan_name, updated_at)
            VALUES ($1, $2, NOW())
            ON CONFLICT (user_id) DO UPDATE
            SET plan_name = $2, updated_at = NOW()
            """,
            user_id, plan_name
        )
        
        return {"status": "success", "message": f"User {user_id} updated to plan: {plan_name}"}

# Version information endpoint
@app.get("/version")
async def version():
    return {
        "version": "1.1.0",
        "features": [
            "Semantic search with pgvector",
            "Keyword boosting",
            "Fallback text search",
            "Debug mode",
            "Bucket listing",
            "Usage tracking"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=API_PORT)
