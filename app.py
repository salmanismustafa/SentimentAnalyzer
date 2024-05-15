from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline


app = Flask(__name__)
CORS(app)

# Load the sentiment analysis model
nlp = pipeline("sentiment-analysis")


@app.route('/sentiment', methods=['POST'])
def analyze_sentiment():
    # Get the input text from the request
    input_text = request.json.get('text')

    # Perform sentiment analysis using the loaded model
    result = nlp(input_text)

    # Extract the sentiment label and score from the result
    sentiment_label = result[0]['label']
    sentiment_score = result[0]['score']

    # Return the sentiment analysis result as JSON
    response_data = {
        'input_text': input_text,
        'sentiment': sentiment_label,
        'confidence_score': sentiment_score
    }
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

