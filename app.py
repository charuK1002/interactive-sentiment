# --- 1. Import all the necessary libraries ---
from flask import Flask, request, jsonify, render_template
import pandas as pd
from transformers import pipeline
import io
import os

# Create a new Flask application
app = Flask(__name__)

# --- 2. Initialize the Hugging Face Sentiment Analysis Pipeline ---
# The model will be loaded only once when the server starts
print("Loading sentiment analysis model...")
try:
    sentiment_pipeline = pipeline("sentiment-analysis")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    # Exit the application if the model fails to load
    exit()

# --- 3. The main page route ---
@app.route('/')
def home():
    """
    Renders the main interactive dashboard HTML page.
    """
    return render_template('index.html')

# --- 4. The API endpoint for analysis ---
@app.route('/analyze', methods=['POST'])
def analyze_data():
    """
    Handles sentiment analysis requests.
    It can process a single text input or an uploaded CSV file.
    """
    try:
        # Check if the request contains a file
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            # Read the CSV file content from the request
            df = pd.read_csv(io.StringIO(file.stream.read().decode('utf-8')))
            
            # Use the "text" column from the CSV
            TEXT_COLUMN = "text"
            
            # Clean the data: drop rows with missing or non-string values
            df = df.dropna(subset=[TEXT_COLUMN])
            df = df[df[TEXT_COLUMN].apply(lambda x: isinstance(x, str))]

            # Perform sentiment analysis on all text at once
            texts = list(df[TEXT_COLUMN])
            if not texts:
                return jsonify({"error": "No valid text found in the CSV file."}), 400
                
            sentiments = sentiment_pipeline(texts)
            
            sentiment_labels = [s['label'] for s in sentiments]
            sentiment_counts = pd.Series(sentiment_labels).value_counts().to_dict()

            return jsonify({"results": sentiment_counts})

        # Check if the request contains single text input
        elif 'text' in request.form:
            text = request.form['text']
            if not text:
                return jsonify({"error": "Text input cannot be empty."}), 400
            
            # Perform sentiment analysis on the single text
            sentiment_result = sentiment_pipeline(text)[0]
            
            return jsonify({"result": sentiment_result['label']})
            
        else:
            return jsonify({"error": "No file or text provided."}), 400

    except KeyError:
        return jsonify({"error": "The specified column 'text' was not found in the CSV file."}), 400
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500

# --- 5. Main entry point ---
if __name__ == '__main__': 
    # The server will run on http://127.0.0.1:5000/
    app.run(debug=True)