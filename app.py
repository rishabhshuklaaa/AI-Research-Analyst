import os
from flask import Flask, request, jsonify
from flask_cors import CORS
# --- IMPORTANT: Import your main class from your other file ---
from final_bot import ResearchAnalystModel

# 1. Initialize Flask App
app = Flask(__name__)
CORS(app) # This allows your UI to make requests to this API

# --- 2. Load API keys and initialize your model ONCE when the server starts ---
print("--- 🚀 Initializing AI Research Analyst Model for the API ---")
my_google_api_key = os.getenv("GOOGLE_API_KEY")
my_news_api_key = os.getenv("NEWS_API_KEY")

# Create a single, shared instance of your model
analyst = ResearchAnalystModel(google_api_key=my_google_api_key, news_api_key=my_news_api_key)
print("--- 🤖 Model Initialized. API is ready to receive requests. ---")


# --- 3. Create API Endpoints for each feature ---

# Endpoint for Data Ingestion
@app.route('/api/ingest', methods=['POST'])
def ingest_endpoint():
    data = request.get_json()
    urls = data.get('urls', [])
    pdfs = data.get('pdfs', []) # Note: PDF upload is more complex, this expects local paths for now
    analyst.ingest_data(urls=urls, pdfs=pdfs)
    return jsonify({"status": "success", "message": "Ingestion process started."})

# Endpoint for Q&A
@app.route('/api/qna', methods=['POST'])
def qna_endpoint():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "Missing 'question' in request body"}), 400
    result = analyst.ask_question(data['question'])
    return jsonify(result)

# Endpoint for SWOT
@app.route('/api/swot', methods=['POST'])
def swot_endpoint():
    data = request.get_json()
    if not data or 'topic' not in data:
        return jsonify({"error": "Missing 'topic' in request body"}), 400
    result = analyst.generate_swot_analysis(data['topic'])
    return jsonify(result)

# Endpoint for Promise Tracker (Compare)
@app.route('/api/compare', methods=['POST'])
def compare_endpoint():
    data = request.get_json()
    if not data or 'topic' not in data or 'source1' not in data or 'source2' not in data:
        return jsonify({"error": "Missing 'topic', 'source1', or 'source2'"}), 400
    result = analyst.track_narrative_evolution(data['topic'], data['source1'], data['source2'])
    return jsonify(result)

# Endpoint for Investment Memo
@app.route('/api/memo', methods=['POST'])
def memo_endpoint():
    data = request.get_json()
    if not data or 'topic' not in data:
        return jsonify({"error": "Missing 'topic' in request body"}), 400
    result = analyst.generate_investment_memo(data['topic'])
    return jsonify(result)

# Endpoint for Market Context
@app.route('/api/context', methods=['POST'])
def context_endpoint():
    data = request.get_json()
    if not data or 'company' not in data or 'competitors' not in data or 'industry' not in data:
        return jsonify({"error": "Missing 'company', 'competitors', or 'industry'"}), 400
    result = analyst.get_market_context(data['company'], data['competitors'], data['industry'])
    return jsonify(result)

# Endpoint for Chart Visualization
@app.route('/api/chart', methods=['POST'])
def chart_endpoint():
    data = request.get_json()
    if not data or 'topic' not in data or 'data_points' not in data:
        return jsonify({"error": "Missing 'topic' or 'data_points'"}), 400
    result_message = analyst.extract_and_visualize_data(data['topic'], data['data_points'])
    # In a real app, you might return the image path or the image file itself
    return jsonify({"status": "success", "message": result_message})

# --- 4. Main block to run the Flask server ---
if __name__ == '__main__':
    # You will run this file: python app.py
    app.run(debug=True, port=5000,use_reloader=False)