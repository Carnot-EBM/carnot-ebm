FROM node:20-slim

# Install Python and dependencies for the bridge
RUN apt-get update && apt-get install -y python3 python3-pip curl && rm -rf /var/lib/apt/lists/*
RUN pip3 install fastapi uvicorn --break-system-packages

# Install Gemini CLI (assuming it's available via npm)
RUN npm install -g @google/gemini-cli

WORKDIR /app
COPY server.py .

# We need to mount ~/.gemini for credentials
ENV GEMINI_BIN=gemini

EXPOSE 8081

CMD ["python3", "server.py"]
