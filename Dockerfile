# Use Python base image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install curl (needed to fetch uv)
RUN apt-get update && apt-get install -y curl

# Install uv and fix permissions
RUN curl -Ls https://astral.sh/uv/install.sh | sh && \
    chmod +x /root/.local/bin/uv

# Add uv to PATH
ENV PATH="/root/.local/bin:$PATH"


# Copy requirements and install with uv
COPY requirements.txt .
#RUN uv pip install -r requirements.txt
RUN uv pip install --system -r requirements.txt


# Copy rest of the code
COPY . .

# Make script executable
RUN chmod +x start.sh

# Expose port
EXPOSE 7860

# Start the app
CMD ["bash", "start.sh"]


# # Install uv
# RUN curl -Ls https://astral.sh/uv/install.sh | sh

# # Add uv to PATH
# ENV PATH="/root/.cargo/bin:$PATH"

# # Upgrade pip (optional since uv wraps pip)
# RUN pip install --upgrade pip  # Optional or remove if using only uv

# # Copy requirements and install dependencies using uv
# COPY requirements.txt .
# RUN pip install -r requirements.txt

# # Copy the rest of your code
# COPY . .

# # Make shell scripts executable
# RUN chmod +x start.sh

# # Expose the port used by your app
# EXPOSE 7860

# # Default command
# CMD ["bash", "start.sh"]
