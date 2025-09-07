import os
import json
import re
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Set the backend before importing pyplot
import matplotlib.pyplot as plt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyMuPDFLoader
from dotenv import load_dotenv
from newspaper import Article
from newsapi import NewsApiClient
from typing import List, Dict

# Load environment variables from .env file
load_dotenv()

class ResearchAnalystModel:
    """
    An interactive AI Research Analyst that ingests data from URLs and PDFs,
    and provides various analytical functions based on user commands.
    """
    def __init__(self, google_api_key: str, news_api_key: str):
        if not google_api_key or not news_api_key:
            raise ValueError("Google and News API Keys are required.")
        os.environ["GOOGLE_API_KEY"] = google_api_key
        self.newsapi = NewsApiClient(api_key=news_api_key)

        self.db_directory = "chroma_db_persistent"
        self.processed_files_file = "processed_files.json"

        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        self.vector_store = Chroma(
            persist_directory=self.db_directory,
            embedding_function=self.embeddings
        )
        print(f"✔️  Vector store loaded. Items in DB: {self.vector_store._collection.count()}")

    def _clean_json_response(self, raw_response: str) -> str:
        """Cleans the raw string output from the LLM to extract a valid JSON substring."""
        match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        return match.group(0) if match else ""

    def _load_processed_files(self) -> set:
        """Loads the set of already processed files."""
        if not os.path.exists(self.processed_files_file): return set()
        with open(self.processed_files_file, 'r') as f:
            try: return set(json.load(f))
            except json.JSONDecodeError: return set()

    def _save_processed_files(self, files: set):
        """Saves the updated set of processed files."""
        with open(self.processed_files_file, 'w') as f:
            json.dump(list(files), f)

    def ingest_data(self, urls: List[str] = None, pdfs: List[str] = None):
        """Ingests data from new URLs and PDFs into the vector store."""
        if urls is None: urls = []
        if pdfs is None: pdfs = []

        processed_files = self._load_processed_files()
        new_urls = [url for url in urls if url not in processed_files]
        new_pdfs = [path for path in pdfs if path not in processed_files]

        if not new_urls and not new_pdfs:
            print("✔️  No new documents to ingest. Database is up to date.")
            return

        documents = []
        # URL Ingestion
        if new_urls:
            print(f"\n📥 Ingesting {len(new_urls)} new URL(s)...")
            for url in new_urls:
                try:
                    article = Article(url)
                    article.download()
                    article.parse()
                    if article.text:
                        doc = Document(page_content=article.text, metadata={"source": url})
                        documents.append(doc)
                except Exception: pass
        
        # PDF Ingestion
        if new_pdfs:
            print(f"\n📥 Ingesting {len(new_pdfs)} new PDF(s)...")
            for file_path in new_pdfs:
                try:
                    loader = PyMuPDFLoader(file_path)
                    pdf_docs = loader.load()
                    for doc in pdf_docs: doc.metadata["source"] = file_path
                    documents.extend(pdf_docs)
                except Exception: pass

        if not documents:
            print("⚠️ Could not create any new documents from the provided sources.")
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        self.vector_store.add_documents(docs)
        
        print(f"✔️  Ingestion complete. Added {len(docs)} new chunks to the database.")
        self._save_processed_files(processed_files.union(set(new_urls)).union(set(new_pdfs)))

   
    def ask_question(self, query: str) -> Dict:
        """Handles general Q&A questions and returns sources."""
        print("\n🤔 Thinking (Q&A)...")
        retriever = self.vector_store.as_retriever()
        # Using a full RAG chain to get context back
        qa_system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the retrieved context to answer the question. If you don't know, say that you don't know."
        "\n\n{context}"
    )
        qa_prompt = ChatPromptTemplate.from_messages([("system", qa_system_prompt), ("human", "{input}")])
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
        response = rag_chain.invoke({"input": query})
    
        # Extract sources from the context that the chain used
        sources = list(set([doc.metadata.get('source', 'unknown') for doc in response.get('context', [])]))
    
        return {"answer": response.get("answer", "No answer found."), "sources": sources}

    def generate_swot_analysis(self, query_context: str) -> Dict:
        """Generates a SWOT analysis."""
        print("\n🤔 Thinking (SWOT Analysis)...")
        swot_prompt = ChatPromptTemplate.from_template(
            "Based on the following context about '{input}', conduct a SWOT analysis. "
            "Present the output as a structured JSON object with keys 'strengths', 'weaknesses', 'opportunities', and 'threats'.\n\n"
            "Context:\n{context}"
        )
        retriever = self.vector_store.as_retriever()
        swot_chain = create_stuff_documents_chain(self.llm, swot_prompt)
        retrieval_chain = create_retrieval_chain(retriever, swot_chain)
        response = retrieval_chain.invoke({"input": query_context})
        try:
            result_json = json.loads(self._clean_json_response(response["answer"]))
            # Add the sources to the final JSON result
            sources = list(set([doc.metadata.get('source', 'unknown') for doc in response.get('context', [])]))
            result_json['sources'] = sources
            return result_json
        except json.JSONDecodeError:
            return {"error": "Failed to generate valid SWOT JSON.", "raw_output": response.get("answer")}

    def track_narrative_evolution(self, topic: str, url1: str, url2: str) -> Dict:
        """Compares two documents to track narrative evolution."""
        print(f"\n🤔 Thinking (Time Machine on '{topic}')...")
        retriever1 = self.vector_store.as_retriever(search_kwargs={"k": 5, "filter": {"source": url1}})
        retriever2 = self.vector_store.as_retriever(search_kwargs={"k": 5, "filter": {"source": url2}})
        docs1, docs2 = retriever1.invoke(topic), retriever2.invoke(topic)

        if not docs1 or not docs2:
             return {"error": "Could not find relevant context for the topic in one or both URLs."}
        context1, context2 = "\n".join([d.page_content for d in docs1]), "\n".join([d.page_content for d in docs2])

        prompt = ChatPromptTemplate.from_template(
            "Analyze the evolution of the narrative on '{topic}' between two documents. "
            "Return a JSON with keys: 'sentiment_change', 'new_information', 'dropped_points', 'summary_of_evolution'.\n\n"
            "Document 1 (Older Context): {context1}\n\nDocument 2 (Newer Context): {context2}"
        )
        chain = prompt | self.llm
        response = chain.invoke({"topic": topic, "context1": context1, "context2": context2})
        try:
            return json.loads(self._clean_json_response(response.content))
        except json.JSONDecodeError:
            return {"error": "Failed to generate valid Evolution JSON.", "raw_output": response.content}

    def generate_investment_memo(self, query_context: str) -> Dict:
        """Generates a one-page investment memo."""
        print("\n🤔 Thinking (Investment Memo)...")
        prompt = ChatPromptTemplate.from_template(
            "Based on the context about '{input}', generate a professional investment memo. "
            "Return a JSON with keys: 'investment_thesis', 'positive_catalysts', 'key_risks', 'conclusion'.\n\n"
            "Context:\n{context}"
        )
        retriever = self.vector_store.as_retriever()
        chain = create_stuff_documents_chain(self.llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, chain)
        response = retrieval_chain.invoke({"input": query_context})
        try:
            return json.loads(self._clean_json_response(response["answer"]))
        except json.JSONDecodeError:
            return {"error": "Failed to generate valid Memo JSON.", "raw_output": response.get("answer")}

    def get_market_context(self, company_name: str, competitors: List[str], industry: str) -> Dict:
        """Fetches and analyzes market and competitor news."""
        print(f"\n🤔 Thinking (Market Context for {company_name})...")
        try:
            news_str = ""
            for item in competitors + [industry]:
                articles = self.newsapi.get_everything(q=item, language='en', sort_by='relevancy', page_size=2)
                news_str += f"\n--- News about {item} ---\n"
                news_str += "\n".join([a['title'] for a in articles['articles']])
            
            prompt = ChatPromptTemplate.from_template(
                "You are a market analyst. Based on the recent news, generate a market context report for '{company_name}'. "
                "Return a JSON with keys: 'overall_sentiment', 'key_competitor_moves', 'major_industry_trends'.\n\nNEWS:\n{news}"
            )
            chain = prompt | self.llm
            response = chain.invoke({"company_name": company_name, "news": news_str})
            return json.loads(self._clean_json_response(response.content))
        except Exception as e:
            return {"error": f"Failed to fetch market news. Error: {e}"}

    def extract_and_visualize_data(self, query_context: str, data_points: List[str]) -> str:
        """Extracts numerical data and generates a chart."""
        print(f"\n🤔 Thinking (Extracting & Visualizing Data)...")
        prompt = ChatPromptTemplate.from_template(
            "From the context about '{input}', extract these data points: {data_points}. "
            "Return a JSON where keys are time periods and values are the data points. "
            "Example: {{\"Q1 2024\": {{\"Sales\": 10000}}, \"Q2 2024\": {{\"Sales\": 12000}}}}.\n\n"
            "Context:\n{context}"
        )
        retriever = self.vector_store.as_retriever()
        chain = create_stuff_documents_chain(self.llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, chain)
        response = retrieval_chain.invoke({"input": f"Extract {', '.join(data_points)} for {query_context}", "data_points": data_points})
        
        try:
            data = json.loads(self._clean_json_response(response["answer"]))
            if not data: return "No data found to visualize."

            df = pd.DataFrame.from_dict(data, orient='index')
            if df.empty: return "Could not structure data for visualization."
            
            df.plot(kind='bar', figsize=(10, 6), grid=True)
            plt.title(f'Financial Data for {query_context}')
            plt.ylabel('Amount')
            plt.xlabel('Time Period')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            chart_filename = "chart.png"
            plt.savefig(chart_filename)
            plt.close()
            return f"✔️  Chart successfully generated and saved as '{chart_filename}'"
        except Exception as e:
            return f"🚨 Failed to visualize data. Error: {e}\nRaw LLM Output:\n{response.get('answer')}"

# --- Main execution block with the interactive workflow ---
if __name__ == "__main__":
    my_google_api_key = os.getenv("GOOGLE_API_KEY")
    my_news_api_key = os.getenv("NEWS_API_KEY")
    
    # --- STAGE 1: SETUP & ONE-TIME DATA INGESTION ---
    print("--- 🚀 Initializing AI Research Analyst ---")
    analyst = ResearchAnalystModel(google_api_key=my_google_api_key, news_api_key=my_news_api_key)

    urls_to_ingest = [
        "https://www.business-standard.com/companies/news/ril-s-q4-results-oil-to-chemicals-business-sees-a-turnaround-in-q4-124042201062.html",
        "https://www.livemint.com/companies/news/reliance-to-invest-rs-60-000-crore-in-green-energy-in-gujarat-11642088864947.html",
        "https://economictimes.indiatimes.com/industry/renewables/reliance-industries-to-commission-new-energy-giga-complex-in-second-half-of-2024/articleshow/107147710.cms",
        "https://economictimes.indiatimes.com/markets/stocks/earnings/bharti-airtel-q4-results-profit-plunges-31-to-rs-2072-crore-arpu-at-rs-209/articleshow/110113115.cms"
    ]
    pdfs_to_ingest = ["TM_report.pdf"]
    
    analyst.ingest_data(urls=urls_to_ingest, pdfs=pdfs_to_ingest)
    
    # --- STAGE 2: INTERACTIVE COMMAND LOOP (MODIFIED STATEMENTS) ---
    
    print("\n\n--- 🤖 AI Research Analyst | Ready for your command ---")
    print("Available commands: 'swot', 'compare', 'memo', 'chart', 'context'.")
    print("Type any other query for Q&A, or 'exit' to quit.")

    while True:
        try:
            user_input = input("\n> Enter command or question: ").lower()

            if user_input in ["exit", "quit"]:
                print("Exiting session. Goodbye!")
                break
            elif "swot" in user_input:
                topic = input("Generate SWOT analysis for which topic? (e.g., Reliance Industries): ")
                result = analyst.generate_swot_analysis(topic)
                print(json.dumps(result, indent=2))
            elif "compare" in user_input or "promise" in user_input:
                print("For Promise Tracker, please provide the following details:")
                topic = input("Topic to track/compare (e.g., Green Energy Strategy): ")
                source1 = input("Enter the first (older) source URL or PDF path: ")
                source2 = input("Enter the second (newer) source URL or PDF path: ")
                result = analyst.track_narrative_evolution(topic, source1, source2)
                print(json.dumps(result, indent=2))
            elif "memo" in user_input:
                topic = input("Generate investment memo for which topic? (e.g., Company's future outlook): ")
                result = analyst.generate_investment_memo(topic)
                print(json.dumps(result, indent=2))
            elif "context" in user_input:
                company = input("Market context for which company?: ")
                competitors_str = input("Enter competitor names (comma-separated): ")
                industry = input("Enter the industry name (e.g., Indian Energy Sector): ")
                result = analyst.get_market_context(company, [c.strip() for c in competitors_str.split(',')], industry)
                print(json.dumps(result, indent=2))
            elif "chart" in user_input or "visualize" in user_input:
                topic = input("Visualize data for which topic? (e.g., Q4 Financial Results): ")
                datapoints_str = input("Enter data points to visualize (comma-separated, e.g., Revenue, Net Profit): ")
                result = analyst.extract_and_visualize_data(topic, [d.strip() for d in datapoints_str.split(',')])
                print(result)
            else:
                result = analyst.ask_question(user_input)
                print(f"\n💡 Answer: {result['answer']}")
        
        except Exception as e:
            print(f"🚨 An error occurred: {e}")