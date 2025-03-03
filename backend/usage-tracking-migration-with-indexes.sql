-- usage-tracking-migration-with-indexes.sql
-- Add tracking tables for Memory Box usage accounting with performance indexes

-- Create plans table
CREATE TABLE IF NOT EXISTS plans (
  id SERIAL PRIMARY KEY,
  name TEXT UNIQUE NOT NULL,
  store_memory_limit INTEGER,
  search_memory_limit INTEGER,
  embedding_limit INTEGER,
  api_call_limit INTEGER,
  storage_limit_bytes INTEGER,
  price_cents INTEGER,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create user_plans table for plan assignment
CREATE TABLE IF NOT EXISTS user_plans (
  user_id TEXT PRIMARY KEY,
  plan_name TEXT NOT NULL DEFAULT 'legacy',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create usage_metrics table for detailed logging
CREATE TABLE IF NOT EXISTS usage_metrics (
  id SERIAL PRIMARY KEY,
  user_id TEXT NOT NULL,
  bucket_id TEXT,
  operation_type TEXT NOT NULL,
  operation_count INTEGER NOT NULL DEFAULT 1,
  bytes_processed INTEGER,
  timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create monthly_usage table for aggregated tracking
CREATE TABLE IF NOT EXISTS monthly_usage (
  id SERIAL PRIMARY KEY,
  user_id TEXT NOT NULL,
  year INTEGER NOT NULL,
  month INTEGER NOT NULL,
  store_memory_count INTEGER NOT NULL DEFAULT 0,
  search_memory_count INTEGER NOT NULL DEFAULT 0,
  embedding_count INTEGER NOT NULL DEFAULT 0,
  api_call_count INTEGER NOT NULL DEFAULT 0,
  total_bytes_processed INTEGER NOT NULL DEFAULT 0,
  UNIQUE(user_id, year, month)
);

-- Insert default plans
INSERT INTO plans (name, store_memory_limit, search_memory_limit, embedding_limit, api_call_limit, storage_limit_bytes, price_cents)
VALUES 
  ('free', 100, 50, 150, 200, 1048576, 0),  -- 1MB storage, limited operations, free
  ('basic', 500, 200, 700, 1000, 10485760, 1000),  -- 10MB storage, $10/month
  ('professional', 2000, 1000, 3000, 5000, 104857600, 2500),  -- 100MB storage, $25/month
  ('legacy', 10000, 5000, 15000, 20000, 1073741824, 0)  -- 1GB storage, generous limits, $0
ON CONFLICT (name) DO NOTHING;

-- Put all existing users on the legacy plan
INSERT INTO user_plans (user_id, plan_name)
SELECT DISTINCT user_id, 'legacy' FROM buckets
ON CONFLICT (user_id) DO NOTHING;

-- Add performance indexes for fast querying

-- Add index on user_id for fast user-specific queries
CREATE INDEX IF NOT EXISTS idx_usage_metrics_user_id ON usage_metrics(user_id);

-- Add index on timestamp for date range queries
CREATE INDEX IF NOT EXISTS idx_usage_metrics_timestamp ON usage_metrics(timestamp);

-- Add composite index for operation type filtering by user
CREATE INDEX IF NOT EXISTS idx_usage_metrics_user_operation ON usage_metrics(user_id, operation_type);

-- Add index on operation_type for system-wide statistics
CREATE INDEX IF NOT EXISTS idx_usage_metrics_operation_type ON usage_metrics(operation_type);

-- Add index on bucket_id for bucket-specific reports
CREATE INDEX IF NOT EXISTS idx_usage_metrics_bucket_id ON usage_metrics(bucket_id);

-- Add indexes for monthly_usage filtering and sorting
CREATE INDEX IF NOT EXISTS idx_monthly_usage_year_month ON monthly_usage(year, month);

-- Analyze the tables to update statistics for the query planner
ANALYZE usage_metrics;
ANALYZE monthly_usage;
