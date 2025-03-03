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
import json
import time
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable

import logging

# Get logger from main application
logger = logging.getLogger("usage_tracking")

class RequestBodyCaptureMiddleware:
    """Middleware to capture request body for later use in the pipeline"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        
        # Create a modified receive function to capture the body
        async def receive_wrapper():
            message = await receive()
            if message["type"] == "http.request":
                body = message.get("body", b"")
                # Store body in scope state for later middleware access
                if "state" not in scope:
                    scope["state"] = {}
                scope["state"]["request_body"] = body
                
                # Don't modify the original message
                return message
            return message
        
        # Pass modified receive function
        await self.app(scope, receive_wrapper, send)

class PoolAccessMiddleware:
    """Middleware to provide database pool access for other middlewares"""
    
    def __init__(self, app):
        self.app = app
        self.pool = None
        
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        
        # Import at runtime to avoid circular imports
        import sys
        import importlib
        main_module = importlib.import_module("main")
        
        # Attach pool to scope state
        if "state" not in scope:
            scope["state"] = {}
        scope["state"]["pool"] = main_module.pool
        
        # Continue processing by passing control to the next middleware
        await self.app(scope, receive, send)

class UsageTrackingMiddleware:
    """Middleware to track API usage metrics for billing purposes"""
    
    def __init__(self, app):
        self.app = app
        
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        
        # We need to process the request normally first
        original_send = send
        response_status = {}
        
        # Create a send wrapper to track response completion
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                response_status["status"] = message.get("status")
            
            # Pass the message to the original send
            await original_send(message)
            
            # After response is complete, track usage asynchronously
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                if response_status.get("status") and response_status["status"] < 500:
                    asyncio.create_task(self._track_usage(scope))
        
        # Continue with the request but use our modified send
        await self.app(scope, receive, send_wrapper)
    
    async def _track_usage(self, scope):
        """Track API usage after response is complete"""
        try:
            # Get pool from scope state
            pool = scope.get("state", {}).get("pool")
            if not pool:
                logger.warning("No database pool available for usage tracking")
                return
                
            # Get URL path and method
            path = scope.get("path", "")
            method = scope.get("method", "")
            
            # Skip non-API routes
            if not path.startswith("/api/"):
                return
                
            # Extract authorization header to get user token
            token = None
            for name, value in scope.get("headers", []):
                if name == b"authorization":
                    auth = value.decode("utf-8")
                    if auth.startswith("Bearer "):
                        token = auth.replace("Bearer ", "")
                        break
            
            if not token:
                return
                
            # Determine operation type
            operation = self._determine_operation(path, method, scope)
            if not operation:
                return
                
            # Record usage asynchronously
            asyncio.create_task(
                self._safe_record_usage(
                    pool=pool,
                    user_id=token,
                    operation=operation,
                    scope=scope
                )
            )
                
        except Exception as e:
            # Log error but don't disrupt response
            logger.error(f"Error in usage tracking: {str(e)}", exc_info=True)
    
    def _determine_operation(self, path: str, method: str, scope: Dict) -> Optional[str]:
        """Map path and method to an operation type for tracking"""
        query_string = scope.get("query_string", b"").decode("utf-8")
        
        # Map endpoints to operation types
        if path == "/api/v2/memory" and method == "POST":
            return "store_memory"
        elif path == "/api/v2/memory" and method == "GET" and "query=" in query_string:
            return "search_memory"
        elif path == "/api/v2/memory" and method == "GET":
            return "api_call"
        elif path == "/api/v2/buckets" and method == "GET":
            return "api_call"
        
        return None
    
    async def _safe_record_usage(self, pool, user_id: str, operation: str, scope: Dict):
        """Safely record usage metrics to database with error handling"""
        try:
            # Get DB connection from pool
            async with pool.acquire() as conn:
                # First ensure tables exist by checking
                tables_exist = await self._check_tables_exist(conn)
                if not tables_exist:
                    logger.warning("Usage tracking tables not yet available, skipping metrics")
                    return
                
                # Ensure user plan exists
                await self._ensure_user_plan(user_id, conn)
                
                # Extract request details
                bucket_id = None
                bytes_processed = None
                
                # Different handling based on operation
                if operation == "store_memory":
                    try:
                        # Try to get request body from scope state
                        request_body = scope.get("state", {}).get("request_body")
                        if request_body:
                            body = json.loads(request_body.decode("utf-8"))
                            bucket_id = body.get("bucketId")
                            text = body.get("text", "")
                            bytes_processed = len(text.encode('utf-8'))
                    except Exception as e:
                        logger.error(f"Error accessing request body: {str(e)}")
                
                elif operation == "search_memory":
                    # Parse query string for bucketId
                    query_string = scope.get("query_string", b"").decode("utf-8")
                    query_params = {}
                    for param in query_string.split("&"):
                        if "=" in param:
                            key, value = param.split("=", 1)
                            query_params[key] = value
                    bucket_id = query_params.get("bucketId")
                
                # Insert into usage_metrics
                await conn.execute(
                    """
                    INSERT INTO usage_metrics
                    (user_id, bucket_id, operation_type, bytes_processed)
                    VALUES ($1, $2, $3, $4)
                    """,
                    user_id, bucket_id, operation, bytes_processed
                )
                
                # Update monthly aggregates
                now = datetime.now()
                await conn.execute(
                    """
                    INSERT INTO monthly_usage
                    (user_id, year, month, store_memory_count, search_memory_count, embedding_count, api_call_count, total_bytes_processed)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (user_id, year, month) DO UPDATE SET
                        store_memory_count = monthly_usage.store_memory_count + $4,
                        search_memory_count = monthly_usage.search_memory_count + $5,
                        embedding_count = monthly_usage.embedding_count + $6,
                        api_call_count = monthly_usage.api_call_count + $7,
                        total_bytes_processed = monthly_usage.total_bytes_processed + $8
                    """,
                    user_id, now.year, now.month,
                    1 if operation == "store_memory" else 0,
                    1 if operation == "search_memory" else 0,
                    1 if operation == "generate_embedding" else 0,
                    1 if operation == "api_call" else 0,
                    bytes_processed or 0
                )
                
                logger.debug(f"Recorded usage: {operation} for user {user_id[:8]}")
        except Exception as e:
            # Log but don't raise - we want the main flow to continue
            logger.error(f"Error recording usage metrics: {str(e)}", exc_info=True)
    
    async def _check_tables_exist(self, conn) -> bool:
        """Check if required tables exist"""
        try:
            result = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'usage_metrics'
                );
                """
            )
            return result
        except Exception:
            return False
    
    async def _ensure_user_plan(self, user_id: str, conn):
        """Ensure a user has a plan record, create if not exists"""
        try:
            plan = await conn.fetchrow(
                "SELECT * FROM user_plans WHERE user_id = $1", 
                user_id
            )
            
            if not plan:
                # Create default legacy plan for existing user
                await conn.execute(
                    "INSERT INTO user_plans(user_id, plan_name) VALUES($1, $2)",
                    user_id, "legacy"
                )
                logger.info(f"Created new plan record for user", extra={"user_id": user_id[:8], "plan": "legacy"})
        except Exception as e:
            logger.error(f"Error ensuring user plan: {str(e)}")
