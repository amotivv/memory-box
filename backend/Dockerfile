FROM python:3.10-slim

# Use ARG to accept build-time variables
ARG ENV=production

# Set environment variable based on ARG
ENV APP_ENV=${ENV}

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Make the start script executable
RUN chmod +x start.sh

# Use the script as the entrypoint
CMD ["./start.sh"]
