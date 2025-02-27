from fastapi import FastAPI, HTTPException, Depends, Header, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union
import os
import httpx
import time
import asyncpg
import logging
import json
import sys
from datetime import datetime

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

app = FastAPI()

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

pool = None

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

# API endpoints
@app.post("/api/v2/memory")
async def add_memory(memory: Memory, authorization: str = Header(None)):
    logger.info("Adding new memory", extra={"bucket_id": memory.bucketId})
    
    # Extract token
    token = authorization.replace("Bearer ", "") if authorization else None
    if not token:
        logger.warning("Unauthorized memory add attempt - missing token")
        raise HTTPException(status_code=401, detail="Missing token")
        
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
    authorization: str = Header(None),
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
    
    # Extract token
    token = authorization.replace("Bearer ", "") if authorization else None
    if not token:
        logger.warning("Unauthorized memory retrieval attempt - missing token")
        raise HTTPException(status_code=401, detail="Missing token")
        
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

# Version information endpoint
@app.get("/version")
async def version():
    return {
        "version": "1.1.0",
        "features": [
            "Semantic search with pgvector",
            "Keyword boosting",
            "Fallback text search",
            "Debug mode"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=API_PORT)
