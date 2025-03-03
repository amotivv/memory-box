# Memory Box Usage Tracking Implementation

This document outlines the implementation of usage tracking for Memory Box to support future billing features.

## Database Structure

### Tables

1. **plans**
   - Defines available service tiers and their limits
   - Contains pricing information and resource allowances
   - Default plans: free, basic, professional, and legacy

2. **user_plans**
   - Maps users to their assigned plan
   - Existing users are automatically assigned to the 'legacy' plan
   - Tracks when a user was assigned to a plan

3. **usage_metrics**
   - Detailed log of individual API operations
   - Records user, bucket, operation type, and data volume
   - Timestamped to support time-based reporting

4. **monthly_usage**
   - Aggregated monthly statistics for each user
   - Categorizes operations by type (store_memory, search_memory, etc.)
   - Maintains running totals for billing calculations

### Indexes

The following indexes have been added to optimize query performance:

1. **usage_metrics table**
   - `idx_usage_metrics_user_id` - Fast lookup by user ID
   - `idx_usage_metrics_timestamp` - Efficient date range queries
   - `idx_usage_metrics_user_operation` - Filtering by user and operation type
   - `idx_usage_metrics_operation_type` - System-wide operation statistics
   - `idx_usage_metrics_bucket_id` - Bucket-specific reports

2. **monthly_usage table**
   - `idx_monthly_usage_year_month` - Fast filtering by time period
   - Unique constraint on (user_id, year, month) - Prevents duplicate entries

## Tracking Implementation

Usage tracking is implemented via ASGI middleware in the FastAPI application:

1. **RequestBodyCaptureMiddleware**
   - Captures request bodies for API calls
   - Stores request body in scope state for later analysis

2. **PoolAccessMiddleware**
   - Provides database connection pool to other middlewares
   - Enables asynchronous database operations

3. **UsageTrackingMiddleware**
   - Records API operations after successful response
   - Categorizes operations based on endpoint and method
   - Extracts operation details (bucket, data size, etc.)
   - Records metrics asynchronously to avoid impacting performance

## Operation Types

The system tracks the following operation types:

- **store_memory** - Adding new memories (POST /api/v2/memory)
- **search_memory** - Semantic search operations (GET /api/v2/memory?query=...)
- **api_call** - Other API calls (bucket listing, etc.)

## Billing Support

The usage tracking system enables various billing models:

1. **Tiered Plans**
   - Plans table defines usage limits for different tiers
   - Monthly aggregation supports billing cycles

2. **Usage-Based Pricing**
   - Track operation counts for metered billing
   - Measure data volume for storage-based pricing

3. **Plan Enforcement**
   - Infrastructure for limit checking (not currently enforced)
   - Legacy users have generous limits with no billing

4. **Analytics**
   - Usage patterns can be analyzed by user, bucket, or operation
   - Time-based reporting for usage trends

## Adding New Plans

To add a new plan, insert a record into the `plans` table:

```sql
INSERT INTO plans (
  name, store_memory_limit, search_memory_limit, 
  embedding_limit, api_call_limit, storage_limit_bytes, price_cents
) VALUES (
  'enterprise', 10000, 5000, 15000, 20000, 10737418240, 10000
);
```

## Assigning Users to Plans

To change a user's plan:

```sql
UPDATE user_plans 
SET plan_name = 'basic', updated_at = NOW() 
WHERE user_id = 'user_token_here';
```

For new users, the system automatically creates a plan record on first API use.

## Upgrading Usage Tracking

1. **Add new metrics**: Modify `UsageTrackingMiddleware._determine_operation` to categorize new operation types.
2. **New dimensions**: Add columns to the `usage_metrics` and `monthly_usage` tables.
3. **Granular tracking**: Current implementation balances detail vs. storage; adjust as needed.
