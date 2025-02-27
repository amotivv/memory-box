-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create memories table
CREATE TABLE memories (
  id SERIAL PRIMARY KEY,
  text TEXT NOT NULL,
  embedding VECTOR(1024),
  bucket_id TEXT NOT NULL,
  user_id TEXT NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index for vector search
CREATE INDEX ON memories USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Create buckets table
CREATE TABLE buckets (
  id SERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  user_id TEXT NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  UNIQUE(name, user_id)
);

-- Add foreign key to memories
ALTER TABLE memories ADD CONSTRAINT fk_bucket 
  FOREIGN KEY (bucket_id, user_id) REFERENCES buckets(name, user_id);

-- Create function for vector similarity search
CREATE OR REPLACE FUNCTION match_memories(
  query_embedding VECTOR(1024),
  match_threshold FLOAT,
  match_count INT,
  user_id_input TEXT
)
RETURNS TABLE (
  id BIGINT,
  text TEXT,
  bucket_id TEXT,
  created_at TIMESTAMPTZ,
  similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    memories.id,
    memories.text,
    memories.bucket_id,
    memories.created_at,
    1 - (memories.embedding <=> query_embedding) AS similarity
  FROM memories
  WHERE
    memories.user_id = user_id_input AND
    1 - (memories.embedding <=> query_embedding) > match_threshold
  ORDER BY similarity DESC
  LIMIT match_count;
END;
$$;
