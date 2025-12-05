import os
from typing import Optional, Dict, Any, TypedDict

os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import streamlit as st
from elevenlabs import ElevenLabs
from firecrawl import Firecrawl

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END


# STREAMLIT: PAGE SETUP
st.set_page_config(page_title="📰 ➡️ 🎙️ Blog to Podcast Agent", page_icon="🎙️")
st.title("📰 ➡️ 🎙️ Blog to Podcast Agent (LangGraph)")


# STREAMLIT: SIDEBAR – API KEYS
st.sidebar.header("🔑 API Keys")
openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
elevenlabs_key = st.sidebar.text_input("ElevenLabs API Key", type="password")
firecrawl_key = st.sidebar.text_input("Firecrawl API Key", type="password")


# STREAMLIT: MAIN INPUT – BLOG URL
url = st.text_input("Enter Blog URL:", "")


# FIRECRAWL SCRAPER HELPERS
def _extract_text_from_firecrawl(doc: Any) -> Optional[str]:
    """
    Try to be very forgiving about Firecrawl's response shape.
    We try many common patterns before giving up.
    """
    if not isinstance(doc, dict):
        return None

    # 1) Direct top-level keys
    for key in ("markdown", "content", "text", "pageContent"):
        if key in doc and isinstance(doc[key], str) and doc[key].strip():
            return doc[key]

    # 2) 'data' key can be dict or list
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

    # 3) Nested 'document' or 'documents'
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

    # If we get here, we didn't find anything clean
    return None


def scrape_blog(url: str) -> str:
    """
    Scrape the main content of a blog from the given URL using Firecrawl
    and return it as markdown/plain text.
    """
    api_key = os.environ.get("FIRECRAWL_API_KEY")
    if not api_key:
        raise ValueError("FIRECRAWL_API_KEY is not set in environment variables.")

    firecrawl = Firecrawl(api_key=api_key)

    doc = firecrawl.scrape(
        url=url,
        formats=["markdown"],
    )

    content = _extract_text_from_firecrawl(doc)

    if not content:
        content = str(doc)

    return content



# LANGGRAPH: STATE DEFINITION
class PodcastState(TypedDict, total=False):
    url: str
    blog_content: str
    summary: str


# -----------------------------
# BUTTON + MAIN PIPELINE
# -----------------------------

if st.button("🎙️ Generate Podcast", disabled=not all([openai_key, elevenlabs_key, firecrawl_key])):
    if not url.strip():
        st.warning("Please enter a blog URL")
    else:
        with st.spinner("Scraping blog and generating podcast..."):
            try:
                # 1) Set environment variables so SDKs can read keys
                os.environ["OPENAI_API_KEY"] = openai_key
                os.environ["FIRECRAWL_API_KEY"] = firecrawl_key

                # 2) Initialize LLM (GPT-4o)
                llm = ChatOpenAI(
                    model="gpt-4o",
                    temperature=0.3,
                )

                # 3) Build LangGraph workflow

                def scrape_node(state: PodcastState) -> Dict[str, Any]:
                    """Node 1: take URL, return blog_content."""
                    content = scrape_blog(state["url"])
                    return {"blog_content": content}

                def summarize_node(state: PodcastState) -> Dict[str, Any]:
                    """Node 2: take blog_content, return podcast summary script."""
                    blog_content = state["blog_content"]

                    system_message = (
                        "You are a helpful AI that turns blog posts into podcast-ready scripts. "
                        "Create a concise, engaging summary suitable for a spoken podcast.\n\n"
                        "Requirements:\n"
                        "- Maximum length: 2000 characters.\n"
                        "- Conversational, friendly tone.\n"
                        "- Capture the main ideas and logical flow.\n"
                        "- Do NOT include meta commentary, just the final script.\n"
                    )

                    prompt = (
                        f"{system_message}\n\n"
                        f"Here is the blog content:\n\n"
                        f"{blog_content}"
                    )

                    response = llm.invoke(prompt)
                    return {"summary": response.content}

                # Create the graph
                graph_builder = StateGraph(PodcastState)

                graph_builder.add_node("scrape", scrape_node)
                graph_builder.add_node("summarize", summarize_node)

                graph_builder.set_entry_point("scrape")
                graph_builder.add_edge("scrape", "summarize")
                graph_builder.add_edge("summarize", END)

                graph = graph_builder.compile()

                # 4) Run the graph
                final_state: PodcastState = graph.invoke({"url": url})
                summary = final_state.get("summary", "")

                if summary:
                    # 5) ElevenLabs: text → speech
                    client = ElevenLabs(api_key=elevenlabs_key)

                    audio_generator = client.text_to_speech.convert(
                        text=summary,
                        voice_id="JBFqnCBsd6RMkjVDRZzb",
                        model_id="eleven_multilingual_v2",
                    )

                    audio_chunks = []
                    for chunk in audio_generator:
                        if chunk:
                            audio_chunks.append(chunk)
                    audio_bytes = b"".join(audio_chunks)
                    # audio_bytes = b"".join([
                    #     b"abc",
                    #     b"def",
                    #     b"ghi",
                    # ])
                    # audio_bytes = b"abcdefghi"


                    # 6) Show results
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
