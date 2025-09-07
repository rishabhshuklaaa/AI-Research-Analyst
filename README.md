# AI Research Analyst Assistant 🦅

AI Research Analyst is a powerful, user-friendly tool designed for in-depth analysis of documents from various sources. Users can input article URLs or upload PDF documents and then leverage a suite of analytical tools to gain relevant insights from the financial and business domains.
<img width="1920" height="948" alt="Screenshot 2025-09-07 192403" src="https://github.com/user-attachments/assets/3428f9be-2ebf-4cfe-933b-44652ca45518" />

---

## ✨ Features

* **Multi-Source Ingestion:** Load data from web article URLs and directly from uploaded PDF files.
* **Persistent Knowledge Base:** Processes content using LangChain, creates embeddings with Google Gemini, and stores them in a persistent Chroma vector database for swift and effective information retrieval.
* **Conversational Q&A:** Interact with the AI by asking questions about the ingested documents and receive answers along with the source documents.
* **Automated SWOT Analysis:** Automatically generate a SWOT (Strengths, Weaknesses, Opportunities, Threats) analysis on any topic based on the provided context.
* **Narrative Evolution Tracker:** Compare two documents to track how a story, promise, or set of facts has evolved over time.
* **Investment Memo Generation:** Create structured, professional investment memos summarizing key theses, risks, and catalysts.
* **Market Context Analysis:** Fetches real-time news via NewsAPI to provide insights into market trends and competitor activities.
* **Data Visualization:** Extracts numerical data from text and generates charts automatically.

---

## 🛠️ Installation

1.  **Clone this repository to your local machine using:**
    ```bash
    git clone [https://github.com/rishabhshuklaaa/AI-Research-Analyst.git](https://github.com/rishabhshuklaaa/AI-Research-Analyst.git)
    ```

2.  **Navigate to the project directory:**
    ```bash
    cd AI-Research-Analyst
    ```

3.  **Install the required dependencies using pip:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your API keys by creating a `.env` file** in the project root and adding your keys. You will need keys from both Google and NewsAPI.
    ```
    GOOGLE_API_KEY="your_google_api_key_here"
    NEWS_API_KEY="your_news_api_key_here"
    ```

---

## 🚀 Usage/Examples

This project has two parts: a backend API and a frontend UI. You need to run both in separate terminals.

1.  **Start the Backend Server:**
    Open a terminal in the project directory and run the Flask app:
    ```bash
    python app.py
    ```
    This will start the server that handles all the AI processing.

2.  **Run the Streamlit Web App:**
    Open a **second, new terminal** and run the Streamlit app:
    ```bash
    streamlit run user_interface.py
    ```

3.  **Use the Application:**
    * The web app will open in your browser.
    * Use the sidebar to input URLs or upload PDF files.
    * Click "Start Analysis Session" to process the documents.
    * Once processing is complete, you can use the different tabs (Q&A, SWOT, etc.) to perform your analysis.

---

## 📂 Project Structure

* **`user_interface.py`**: The main Streamlit application script for the frontend.
* **`app.py`**: The Flask backend script that serves the API.
* **`final_bot.py`**: Contains the core AI logic, chains, and analytical functions.
* **`requirements.txt`**: A list of all required Python packages.
* **`chroma_db_persistent/`**: The folder where the Chroma vector database is stored.
* **`processed_files.json`**: A checklist to keep track of already ingested files.
* **`.env`**: Configuration file for storing your secret API keys.
