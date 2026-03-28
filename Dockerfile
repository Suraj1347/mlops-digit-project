# 1. Choose the base computer to run your app
FROM python:3.9-slim

# 2. Set the folder where everything will live inside the container
WORKDIR /app

# 3. Copy your requirements file first
COPY requirements.txt .

# 4. Install all your Python libraries (FastAPI, OpenCV, etc.)
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your files (app.py, model.pkl, etc.)
COPY . .

# 6. Expose the port your API will run on
EXPOSE 8000

# 7. Start the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]