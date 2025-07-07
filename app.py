import streamlit as st
import feedparser
import json
import logging
from io import StringIO

from orchestrator import AgentOrchestrator

st.set_page_config(
    page_title="Fintech Agent Monitor",
    page_icon="🤖",
    layout="wide"
)

# --- Helper functions for UI ---
def load_user_config():
    """Loads the user's category preferences from user_config.json."""
    try:
        with open("user_config.json", "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"enabled_categories": []}

def save_user_config(config):
    """Saves the user's category preferences."""
    with open("user_config.json", "w") as f:
        json.dump(config, f, indent=2)

def get_all_rss_categories(rss_url):
    """Parses the RSS feed to get a unique list of all available categories."""
    try:
        feed = feedparser.parse(rss_url)
        all_categories = set()
        for entry in feed.entries:
            if hasattr(entry, 'tags'):
                for tag in entry.tags:
                    all_categories.add(tag.term)
        return sorted(list(all_categories))
    except Exception as e:
        st.error(f"Could not fetch RSS categories: {e}")
        return []
    

# --- UI Layout ---
st.title("Agentic fintech newsletter monitor")
st.markdown("A simple interface to manage and query your AI agent system that monitors 'This Week in Fintech'.")

# --- Sidebar for Settings and Controls ---
with st.sidebar:
    st.header("⚙️ Settings & Controls")
    
    st.subheader("Newsletter Categories")
    st.markdown("Select the topics you want the agent to monitor.")

    # Load current settings and all possible categories
    user_config = load_user_config()
    enabled_categories = user_config.get("enabled_categories", [])
    all_categories = get_all_rss_categories("https://thisweekinfintech.com/rss/")
    
    # Create checkboxes for each category
    new_enabled_categories = []
    for category in all_categories:
        if st.checkbox(category, value=(category in enabled_categories)):
            new_enabled_categories.append(category)

    if st.button("Save Settings"):
        user_config["enabled_categories"] = new_enabled_categories
        save_user_config(user_config)
        st.success("Settings saved successfully!")
        st.rerun() # Rerun to reflect changes immediately

    st.divider()
    st.header("🛠️ Manual Actions")

    # Initialize the orchestrator once
    if 'orchestrator' not in st.session_state:
        st.session_state.orchestrator = AgentOrchestrator()
    
    run_check_button = st.button("Run Daily Check Now", type="primary")
    
    st.warning("The backfill process can take several minutes to complete.")
    run_backfill_button = st.button("Backfill Historical Articles")


# --- Main Page for Status and Q&A ---
log_container = st.expander("Monitoring Logs", expanded=True)
log_stream = StringIO()

# Setup a logger to capture output for the UI
logging.basicConfig(stream=log_stream, level=logging.INFO, force=True, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Action Handling ---
if run_check_button:
    with st.spinner("🔍 Checking for new articles based on your settings..."):
        st.session_state.orchestrator.run_monitoring_cycle()
        log_container.code(log_stream.getvalue())
        st.success("Monitoring cycle complete!")

if run_backfill_button:
    with st.spinner("⏳ Backfilling historical articles... This will take a few minutes."):
        st.session_state.orchestrator.backfill_historical_articles()
        log_container.code(log_stream.getvalue())
        st.success("Historical backfill complete!")

# --- Q&A Section ---
st.divider()
st.header("💬 Ask a Question")
st.markdown("Query the knowledge base built from the newsletters.")

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about recent fintech news..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            response = st.session_state.orchestrator.answer_question(prompt)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})