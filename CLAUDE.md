# Memory Plugin Project Guidelines

## Build/Run Commands
- Start development environment: `docker-compose up -d`
- Start production environment: `APP_ENV=production docker-compose up -d`
- View logs: `docker-compose logs -f memory-api`
- Create bucket: `docker-compose exec memory-api python create_bucket.py "BucketName" "user-id"`
- Rebuild containers: `docker-compose build --no-cache`
- Stop services: `docker-compose down`

## Code Style Guidelines
- **Imports**: Group and order by standard library, third-party, and local modules
- **Types**: Use Python type hints with Pydantic models for API requests/responses
- **Logging**: Use structured logging with context in `extra` parameter
- **Error Handling**: Use FastAPI's HTTPException with appropriate status codes
- **Naming**: Use snake_case for variables/functions, PascalCase for classes
- **Documentation**: Use docstrings for functions and detailed class descriptions
- **API Path Format**: RESTful `/api/v{version}/{resource}` pattern
- **Environment Variables**: Use consistent naming and provide default values

## Project Structure
- Backend: FastAPI (Python), PostgreSQL with pgvector
- Containerized with Docker and Docker Compose
- Uses Ollama API for embeddings