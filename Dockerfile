# Start from a small official Python image
FROM python:3.11-slim

# All app files live in /app inside the box
WORKDIR /app

# Install packages first, so Docker can reuse this step when only code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project (code, data, tests)
COPY . .

# Streamlit's port
EXPOSE 8501

# Start the app when the box runs
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
