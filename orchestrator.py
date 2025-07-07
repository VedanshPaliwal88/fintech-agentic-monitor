import os
import json
import logging
import time
import hashlib
from datetime import datetime
from typing import List, Dict
import argparse
import feedparser
import html2text

from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import chromadb
from chromadb.config import Settings as ChromaSettings 
from sentence_transformers import SentenceTransformer
from groq import Groq
from agno.agent import Agent
from tavily import TavilyClient

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

# --- Application Settings ---
SETTINGS = {
    "base_url": "https://thisweekinfintech.com",
    "rss_feed_url" : "https://thisweekinfintech.com/rss",
    "backfill_tags": {
        "North America": "/tag/us",
        "Asia/India": "/tag/asia",
        "Stablecoins": "/tag/weekly-stable"
    },
    "state_file": "state.json",
    "db_path": "db",
    "embedding_model": "all-MiniLM-L6-v2",
    "llm_model": "llama3-8b-8192",
    "chunk_size_words": 400,
    "request_timeout": 15,
    "retry_attempts": 3,
    "retry_delay_seconds": 5
}

def load_user_config():
    """Loads the user's category preferences from a JSON file."""
    try:
        with open("user_config.json", "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # If the file doesn't exist or is invalid, return a default
        return {"enabled_categories": []}

class AgentOrchestrator:
    def __init__(self):
        logger.info("Initializing Agent Orchestrator...")
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.embedding_model = SentenceTransformer(SETTINGS["embedding_model"])
        
        self.chroma_client = chromadb.PersistentClient(
            path=SETTINGS["db_path"],
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="newsletter_content"
        )
        self.tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        self.html_parser = html2text.HTML2Text()
        self.html_parser.ignore_links = True
        self.html_parser.ignore_images = True

        self._setup_agents()
        logger.info("Orchestrator initialized successfully.")

    def _setup_agents(self):
        """Define the roles and tools for each agent in the system."""
        self.monitor_agent = Agent(
            name="NewsletterMonitor",
            role="Checks for new newsletter articles across specified topics.",
            goal="Identify and report new articles since the last check.",
            tools=[self.check_rss_for_new_articles_tool]
        )
        self.content_agent = Agent(
            name="ContentExtractor",
            role="Extracts and stores content from new articles.",
            goal="Parse article HTML, clean the text, and store it in the vector database.",
            tools=[self.extract_and_store_content_tool]
        )
        self.rag_agent = Agent(
            name="RAGAnswerer",
            role="Answers questions using both an internal knowledge base and live web searches.",
            goal="Provide comprehensive answers by combining stored newsletter content with real-time information.",
            tools=[self.retrieve_context_tool, self.web_search_tool]
        )
        logger.info("Agents have been configured.")

    # --- Core Logic and Tools (Omitted for brevity, they are unchanged) ---
    def _load_state(self) -> Dict:
        try:
            if os.path.exists(SETTINGS["state_file"]):
                with open(SETTINGS["state_file"], "r") as f:
                    return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            pass
        return {"processed_guids": {}}

    def _save_state(self, state: Dict):
        with open(SETTINGS["state_file"], "w") as f:
            json.dump(state, f, indent=4)

    def backfill_historical_articles(self):
        """
        Scrapes all available articles from specific tag pages to populate the database.
        This uses the old web scraping method for historical data not in the RSS feed.
        """
        logger.info("🚀 Starting historical backfill process. This may take a while...")
        
        # Use a dictionary keyed by link to store unique articles, avoiding duplicates
        unique_articles_to_process = {}

        for tag_name, tag_path in SETTINGS["backfill_tags"].items():
            logger.info(f"🔎 Scraping historical articles for tag: {tag_name}")
            try:
                tag_url = f"{SETTINGS['base_url']}{tag_path}"
                response = self._make_request(tag_url)
                if not response:
                    continue

                soup = BeautifulSoup(response.content, 'html.parser')
                # Use find_all to get every article card on the page
                article_cards = soup.find_all("article", class_="gh-card")
                logger.info(f"Found {len(article_cards)} article cards on the page for '{tag_name}'.")

                for card in article_cards:
                    link_element = card.find("a", class_="gh-card-link")
                    if link_element and 'href' in link_element.attrs:
                        link = link_element['href']
                        if link not in unique_articles_to_process:
                            title = (link_element.find("h3", class_="gh-card-title") or link_element).text.strip()
                            unique_articles_to_process[link] = {
                                'tag': tag_name,
                                'title': title,
                                'link': link
                            }

            except Exception as e:
                logger.error(f"Error scraping tag '{tag_name}': {e}", exc_info=True)

        total_found = len(unique_articles_to_process)
        if total_found == 0:
            logger.info("🏁 No historical articles found to process.")
            return

        logger.info(f"Found a total of {total_found} unique historical articles to process.")
        
        for i, article_info in enumerate(unique_articles_to_process.values()):
            logger.info(f"Backfilling article {i+1}/{total_found}: {article_info['title']}")
            # We can reuse our existing, robust tool for extraction and storage!
            self.extract_and_store_content_tool(article_info)
        
        logger.info("✅ Historical backfill process complete.")

    def check_rss_for_new_articles_tool(self):
        '''
        Checks the rss feed for new articles using their GUID and category.
        '''
        logger.info("🔍 Checking RSS feed for new articles...")
        state = self._load_state()

        processed_guids = set(state.get("processed_guids", []))
        new_articles_to_process = []

        user_config = load_user_config()
        allowed_categories = set(user_config.get("enabled_categories", []))

        feed = feedparser.parse(SETTINGS["rss_feed_url"])

        for entry in feed.entries:
            guid = entry.id
            if guid not in processed_guids:
                mathed_category = None
                if hasattr(entry, 'tags'):
                    for tag in entry.tags:
                        if tag.term in allowed_categories:
                            mathed_category = tag.term
                            break
                if mathed_category:
                    logger.info(f"New article found via RSS: {entry.title}")
                    new_articles_to_process.append({
                        'guid': guid,
                        'title': entry.title,
                        'link': entry.link,
                        'pubDate': entry.published,
                        'full_content_html': entry.content[0].value if entry.content else entry.summary
                    })
                else:
                    logger.info(f"Skipping new irrelevant article: {entry.title}")

                processed_guids.add(guid)

        if not new_articles_to_process:
            logger.info("No new articles found in the RSS feed.")

        self._save_state({"processed_guids": list(processed_guids)})
        return new_articles_to_process
    
    def extract_and_store_content_tool(self, article_info: Dict[str, str]):
        '''
        Receives raw content and stores it
        Can receive html through RSS or fetch it via a link
        '''
        try:
            if article_info.get('full_content_html'):
                logger.info(f"Processing content for article: '{article_info['title']}' directly from RSS feed")
                html_content = article_info['full_content_html']
            else:
                article_url = f"{SETTINGS['base_url']}{article_info['link']}"
                logger.info(f"Fetching content for: {article_info['title']} ({article_url})")
                response = self._make_request(article_url)
                if not response: return f"Failed to fetch content for {article_info['title']}."
                soup = BeautifulSoup(response.content, 'html.parser')
                content_container = soup.find("article", class_="gh-article")
                if not content_container:
                    return f"Content container not found for {article_info['title']}."
                html_content = str(content_container)

            plain_text = self.html_parser.handle(html_content)
            chunks = self._chunk_text(plain_text)

            article_id = hashlib.sha256(article_info['link'].encode()).hexdigest()

            if self._store_chunks(article_id, article_info, chunks):
                return f"Successfully extracted and stored {len(chunks)} chunks from '{article_info['title']}."
            else:
                return "No content to store."
    
        except Exception as e:
            logger.error(f"Error extracting/storing '{article_info['title']}': {e}", exc_info=True)
            return f"An error occured during extraction: {e}"
        
    def _store_chunks(self, article_id: str, article_info: Dict, chunks: List[str]) -> bool:
        '''
        Helper to handle vector db storage logic
        '''
        if not chunks:
            return False

        embeddings, documents, metadatas, ids = [], [], [], []
        for i, chunk in enumerate(chunks):
            enhanced_chunk = f"Article Title: {article_info['title']}\nTag: {article_info.get('tag', 'N/A')}\n\n{chunk}"
            documents.append(enhanced_chunk)
            metadatas.append({
                "title": article_info['title'],
                "tag": article_info.get('tag', 'N/A'),
                "timestamp": article_info.get('pubDate', datetime.now().isoformat()),
                "link": article_info['link'],
                "chunk_id": i
            })
            ids.append(f"{article_id}_{i}")

        if documents:
            embeddings_list = self.embedding_model.encode(documents).tolist()
            self.collection.add(embeddings=embeddings_list, documents=documents, metadatas=metadatas, ids=ids)
            logger.info(f"💾 Stored {len(chunks)} chunks for '{article_info['title']}' in the vector database.")
            return True
        return False


    def retrieve_context_tool(self, query: str, n_results: int = 5):
        """Searches the internal vector DB for context relevant to the query."""
        logger.info(f"📚 Searching internal database for: '{query}'")
        try:
            query_embedding = self.embedding_model.encode([query]).tolist()
            results = self.collection.query(query_embeddings=query_embedding, n_results=n_results)
            return results['documents'][0] if results['documents'] else []
        except Exception as e:
            logger.error(f"Error searching content: {e}", exc_info=True)
            return []

    def web_search_tool(self, query: str) -> str:
        """
        Performs a live web search for a given query to get up-to-date information.
        """
        logger.info(f"🌐 Performing live web search for: '{query}'")
        try:
            response = self.tavily_client.search(query=query, search_depth="basic")
            # Format the results into a readable string
            result_string = "\n".join([f"- {res['content']}" for res in response['results']])
            return f"Web search results for '{query}':\n{result_string}"
        except Exception as e:
            logger.error(f"Error performing web search: {e}")
            return "Web search failed."

    def run_monitoring_cycle(self):
        '''Daily monitoring for updates using RSS'''
        logger.info("🚀 Starting new monitoring cycle using RSS feed...")
        new_articles = self.check_rss_for_new_articles_tool()

        if not new_articles:
            return
        
        logger.info(f"Found {len(new_articles)} new articles to process.")

        processing_summaries = [self.extract_and_store_content_tool(article) for article in new_articles]
        
        final_summary = f"🏁 Monitoring cycle complete.\nProcessed {len(new_articles)} new articles.\n" + "\n".join(processing_summaries)
        logger.info(final_summary)

    def answer_question(self, question: str):
        """
        Answers a question using a Hybrid RAG approach:
        1. Retrieves context from the local newsletter DB.
        2. Retrieves context from a live web search.
        3. Synthesizes an answer from the combined context.
        """
        # Step 1: Retrieve from internal database (using the RAG agent's tool)
        internal_context_list = self.rag_agent.tools[0](query=question) # retrieve_context_tool
        internal_context = "\n\n".join(internal_context_list)

        # Step 2: Retrieve from live web search (using the RAG agent's other tool)
        web_context = self.rag_agent.tools[1](query=question) # web_search_tool

        # Step 3: Create the final prompt with both contexts
        prompt = f"""
            **Role:** You are a helpful and knowledgeable fintech analyst.
            **Task:** Answer the user's question by synthesizing information from both internal newsletters and live web search results.

            **Instructions:**
            1.  Prioritize information from the "Newsletter Context" as it is from a trusted, specific source.
            2.  Use the "Live Web Search Results" to supplement, verify, or provide more recent information if the newsletters are outdated or lack detail.
            3.  Synthesize a single, coherent answer. Do not simply list the different sources.
            4.  If the sources conflict, acknowledge the discrepancy.
            5. Always specify the newsletter title and the sources used for the information
            6.  If neither source contains the necessary information, state that. Do not use external knowledge.

            **Newsletter Context:**
            ---
            {internal_context if internal_context else "No relevant information found in newsletters."}
            ---

            **Live Web Search Results:**
            ---
            {web_context if web_context else "No relevant information found from web search."}
            ---

            **User's Question:** {question}

            **Answer:**
            """

        try:
            logger.info("Generating answer with Groq LLM using combined context...")
            response = self.groq_client.chat.completions.create(
                model=SETTINGS["llm_model"], 
                messages=[{"role": "user", "content": prompt}], 
                temperature=0.2
            )
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"Error generating answer with LLM: {e}", exc_info=True)
            return "I encountered an error while trying to generate an answer. Please try again."

    def _make_request(self, url: str):
        for attempt in range(SETTINGS["retry_attempts"]):
            try:
                response = requests.get(url, timeout=SETTINGS["request_timeout"])
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request to {url} failed (attempt {attempt+1}/{SETTINGS['retry_attempts']}): {e}")
                if attempt < SETTINGS["retry_attempts"] - 1: time.sleep(SETTINGS["retry_delay_seconds"])
                else: logger.error(f"Failed to fetch {url} after {SETTINGS['retry_attempts']} attempts.")
        return None

    def _chunk_text(self, content: str):
        words = content.split()
        return [" ".join(words[i:i + SETTINGS["chunk_size_words"]]) for i in range(0, len(words), SETTINGS["chunk_size_words"])]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Agentic Newsletter Monitor for 'This Week in Fintech'."
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Run the historical backfill process to populate the database with all available articles."
    )
    args = parser.parse_args()

    orchestrator = AgentOrchestrator()

    if args.backfill:
        orchestrator.backfill_historical_articles()

    else:
        logger.info("\n" + "="*50 + "\nRUNNING INITIAL MONITORING CYCLE\n" + "="*50)
        orchestrator.run_monitoring_cycle()

        print("\n" + "="*50)
        print("INTERACTIVE Q&A MODE - type 'quit' to exit")
        print("="*50)
        while True:
            try:
                question = input("\nYour question: ")
                if question.lower().strip() == 'quit': break
                if not question: continue
                answer = orchestrator.answer_question(question)
                print(f"\n🤖 AI Answer:\n{answer}")
            except (KeyboardInterrupt, EOFError):
                print("\nExiting...")
                break