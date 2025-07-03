# Agentic AI Fintech Newsletter Monitor

**Agentic AI Fintech Newsletter Monitor** is an autonomous AI system that tracks the *This Week in Fintech* newsletter, extracts key insights, and answers user queries using a powerful hybrid RAG (Retrieval-Augmented Generation) approach.

---

## Key Features

- **Automated Daily Monitoring**  
  Automatically checks for new newsletter content every day.

- **Intelligent Content Extraction**  
  Parses and processes new articles, converting them into vector embeddings for efficient retrieval.

- **Hybrid Q&A System**  
  Answers user questions by combining:
  - A local knowledge base of historical newsletters.
  - Live web search results for the latest information.

- **Persistent Memory**  
  Uses **ChromaDB** to persist data on disk, ensuring knowledge is retained across restarts.

---

## System Architecture

The system is composed of three autonomous agents:

### Monitor Agent
Scans the newsletter website and detects newly published articles that haven’t been indexed yet.

### Content Agent
Fetches and scrapes content from new article links, cleans the data, and stores embeddings in ChromaDB.

### RAG Agent
Handles user queries by:
- Retrieving relevant information from ChromaDB.
- Performing a live web search.
- Synthesizing an answer from both sources.

---

## Project Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/VedanshPaliwal88/fintech-agentic-monitor.git
   cd fintech-agentic
2. **Create a Virtual Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate # or venv\Scripts\activate on Windows
3. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
4. **Configure Environment Variables**
    - Copy .env.example to .env
    - Update values as needed

5. **Running the System**

   #### 5.1 One-time Database Back-fill (Recommended First Step)

   To give your agent a deep historical context, you should first run the backfill process. This will scrape all available articles from the configured tag pages and load them into the vector database.

   > **Warning:** This may take several minutes to complete.

   Run the following command:

   ```bash
   python orchestrator.py --backfill
   ```

   #### 5.2 To Run the Main System (Subsequent Accesses)

   After the initial backfill, run the main orchestrator to start the autonomous system:

   ```bash
   python orchestrator.py
   ```



## Future Work & Improvements
- Web UI: Create a simple web interface (using Streamlit or Flask) for a more user-friendly Q&A experience.
- Email Notifications: Add functionality to send an email summary when new articles are found and processed.
- Expanded Data Sources: Add more newsletters or blogs for the agents to monitor.