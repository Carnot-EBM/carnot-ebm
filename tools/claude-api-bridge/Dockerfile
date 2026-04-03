# Claude Code API Bridge
# Wraps `claude -p` CLI as an OpenAI-compatible API server.
#
# Build:
#   docker build -t claude-api-bridge .
#
# Run (mount your Claude OAuth credentials):
#   docker run -v ~/.claude:/root/.claude -p 8080:8080 claude-api-bridge
#
# Then use any OpenAI SDK pointed at http://localhost:8080/v1

FROM node:22-slim

# Install Python 3 + pip for the FastAPI server
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install Claude Code CLI globally
RUN npm install -g @anthropic-ai/claude-code

# Create a Python virtual environment (required on Debian/Ubuntu)
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi>=0.115.0 \
    uvicorn[standard]>=0.34.0

# Copy the server code
WORKDIR /app
COPY server.py .

# Claude stores OAuth tokens in ~/.claude
# Mount this from the host: -v ~/.claude:/root/.claude
VOLUME ["/root/.claude"]

# The server listens on port 8080
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Start the API bridge server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
