import os
from typing import Optional, Dict, Any, TypedDict

# Load .env (for local development)
from dotenv import load_dotenv
load_dotenv()

# Fix Streamlit watcher issue
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import streamlit as st
from elevenlabs import ElevenLabs
from firecrawl import Firecrawl
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END


# -----------------------------
# STREAMLIT: PAGE SETUP
# -----------------------------
st.set_page_config(page_title="📰 ➡️ 🎙️ Blog to Podcast Agent", page_icon="🎙️")
st.title("📰 ➡️ 🎙️ Blog to Podcast Agent (LangGraph)")


# -----------------------------
# CHECK REQUIRED ENV VARIABLES
# -----------------------------
required_keys = ["OPENAI_API_KEY", "ELEVENLABS_API_KEY", "FIRECRAWL_API_KEY"]
missing_keys = [k for k in required_keys if not os.environ.get(k)]

if missing_keys:
    st.error(f"❌ Missing environment variables: {', '.join(missing_keys)}")
    st.info("Create a .env file or export them in your terminal.")
    st.stop()


# -----------------------------
# INPUT: BLOG URL
# -----------------------------
url = st.text_input("Enter Blog URL:", "")


# -----------------------------
# FIRECRAWL SCRAPER HELPERS
# -----------------------------
def _extract_text_from_firecrawl(doc: Any) -> Optional[str]:
    if not isinstance(doc, dict):
        return None

    for key in ("markdown", "content", "text", "pageContent"):
        if key in doc and isinstance(doc[key], str) and doc[key].strip():
            return doc[key]

    data = doc.get("data")
    if isinstance(data, dict):
        for key in ("markdown", "content", "text", "pageContent"):
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                return val

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                for key in ("markdown", "content", "text", "pageContent"):
                    val = item.get(key)
                    if isinstance(val, str) and val.strip():
                        return val

    document = doc.get("document")
    if isinstance(document, dict):
        for key in ("markdown", "content", "text", "pageContent"):
            val = document.get(key)
            if isinstance(val, str) and val.strip():
                return val

    documents = doc.get("documents")
    if isinstance(documents, list):
        for item in documents:
            if isinstance(item, dict):
                for key in ("markdown", "content", "text", "pageContent"):
                    val = item.get(key)
                    if isinstance(val, str) and val.strip():
                        return val

    return None


def scrape_blog(url: str) -> str:
    api_key = os.environ.get("FIRECRAWL_API_KEY")
    firecrawl = Firecrawl(api_key=api_key)

    doc = firecrawl.scrape(url=url, formats=["markdown"])
    content = _extract_text_from_firecrawl(doc)

    return content if content else str(doc)


# -----------------------------
# LANGGRAPH STATE
# -----------------------------
class PodcastState(TypedDict, total=False):
    url: str
    blog_content: str
    summary: str


# -----------------------------
# MAIN BUTTON
# -----------------------------
if st.button("🎙️ Generate Podcast"):
    if not url.strip():
        st.warning("Please enter a blog URL")
    else:
        with st.spinner("Scraping blog and generating podcast..."):
            try:
                # LLM
                llm = ChatOpenAI(
                    model="gpt-4o",
                    temperature=0.3,
                )

                # Nodes
                def scrape_node(state: PodcastState) -> Dict[str, Any]:
                    content = scrape_blog(state["url"])
                    return {"blog_content": content}

                def summarize_node(state: PodcastState) -> Dict[str, Any]:
                    blog_content = state["blog_content"]

                    prompt = f"""
You are a helpful AI that turns blog posts into podcast-ready scripts.

Requirements:
- Max 2000 characters
- Conversational tone
- No meta commentary

Blog:
{blog_content}
"""

                    response = llm.invoke(prompt)
                    return {"summary": response.content}

                # Graph
                graph_builder = StateGraph(PodcastState)
                graph_builder.add_node("scrape", scrape_node)
                graph_builder.add_node("summarize", summarize_node)

                graph_builder.set_entry_point("scrape")
                graph_builder.add_edge("scrape", "summarize")
                graph_builder.add_edge("summarize", END)

                graph = graph_builder.compile()

                # Run
                final_state = graph.invoke({"url": url})
                summary = final_state.get("summary", "")

                if summary:
                    # TTS
                    client = ElevenLabs(
                        api_key=os.environ.get("ELEVENLABS_API_KEY")
                    )

                    audio_generator = client.text_to_speech.convert(
                        text=summary,
                        voice_id="JBFqnCBsd6RMkjVDRZzb",
                        model_id="eleven_multilingual_v2",
                    )

                    audio_bytes = b"".join([chunk for chunk in audio_generator if chunk])

                    # Output
                    st.success("Podcast generated! 🎧")
                    st.audio(audio_bytes, format="audio/mp3")

                    st.download_button(
                        "Download Podcast",
                        audio_bytes,
                        "podcast.mp3",
                        "audio/mp3",
                    )

                    with st.expander("📄 Podcast Summary"):
                        st.write(summary)

                else:
                    st.error("Failed to generate summary")

            except Exception as e:
                st.error(f"Error: {e}")