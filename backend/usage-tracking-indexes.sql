--
-- Memory Box - A semantic memory storage and retrieval system
-- Copyright (C) 2025 amotivv, inc.
--
-- This program is free software: you can redistribute it and/or modify
-- it under the terms of the GNU Affero General Public License as published by
-- the Free Software Foundation, either version 3 of the License, or
-- (at your option) any later version.
--
-- This program is distributed in the hope that it will be useful,
-- but WITHOUT ANY WARRANTY; without even the implied warranty of
-- MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
-- GNU Affero General Public License for more details.
--
-- You should have received a copy of the GNU Affero General Public License
-- along with this program.  If not, see <https://www.gnu.org/licenses/>.
--

-- usage-tracking-indexes.sql
-- Add performance indexes to usage tracking tables

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
