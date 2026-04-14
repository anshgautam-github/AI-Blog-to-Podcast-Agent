import os
from typing import Dict, Any, TypedDict
from dotenv import load_dotenv

# Load env
load_dotenv()

os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import streamlit as st
import requests
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
from elevenlabs import ElevenLabs
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Blog → Podcast Agent", page_icon="🎙️")
st.title("📰 ➡️ 🎙️ Blog to Podcast Agent (Smart)")

url = st.text_input("Enter Blog or YouTube URL:")


# -----------------------------
# ENV CHECK
# -----------------------------
required_keys = ["OPENAI_API_KEY", "ELEVENLABS_API_KEY"]
missing = [k for k in required_keys if not os.environ.get(k)]

if missing:
    st.error(f"Missing env vars: {', '.join(missing)}")
    st.stop()


# -----------------------------
# SCRAPERS
# -----------------------------



def get_youtube_transcript(url: str) -> str:
    try:
        video_id = url.split("v=")[-1].split("&")[0]

        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id)

        return " ".join([t.text for t in transcript])

    except Exception as e:
        return f"Error fetching transcript: {str(e)}"

def scrape_webpage(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers, timeout=10)

    soup = BeautifulSoup(res.text, "html.parser")

    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    clean = "\n".join(line for line in lines if line)

    return clean[:10000]


def extract_content(url: str) -> str:
    if "youtube.com" in url or "youtu.be" in url:
        return get_youtube_transcript(url)
    else:
        return scrape_webpage(url)


# -----------------------------
# LANGGRAPH STATE
# -----------------------------
class State(TypedDict):
    url: str
    content: str
    summary: str


# -----------------------------
# MAIN BUTTON
# -----------------------------
if st.button("🎙️ Generate Podcast"):
    if not url.strip():
        st.warning("Enter a valid URL")
    else:
        with st.spinner("Processing..."):
            try:
                llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

                def scrape_node(state: State) -> Dict[str, Any]:
                    return {"content": extract_content(state["url"])}

                def summarize_node(state: State) -> Dict[str, Any]:
                    prompt = f"""
                                    You are a professional podcast writer.

                                    Your task:
                                    Convert the given content into a clean, standalone podcast script.

                                    STRICT RULES:
                                    - Remove references to videos, channels, or “next/previous video”
                                    - Remove phrases like “subscribe”, “like”, “comment”
                                    - Do NOT mention YouTube or that this came from a video
                                    - Do NOT say “in this video”
                                    - Rewrite content as a smooth, self-contained narration
                                    - Make it sound like a podcast host speaking to listeners
                                    - Add a natural intro and closing line
                                    - Keep it conversational and engaging
                                    - Max 2000 characters

                                    CONTENT:
                                    {state['content']}
                            """
                    res = llm.invoke(prompt)
                    return {"summary": res.content}

                graph = StateGraph(State)

                graph.add_node("scrape", scrape_node)
                graph.add_node("summarize", summarize_node)

                graph.set_entry_point("scrape")
                graph.add_edge("scrape", "summarize")
                graph.add_edge("summarize", END)

                app = graph.compile()

                result = app.invoke({"url": url})
                summary = result.get("summary", "")

                if summary:
                    client = ElevenLabs(
                        api_key=os.environ.get("ELEVENLABS_API_KEY")
                    )

                    audio = client.text_to_speech.convert(
                        text=summary,
                        voice_id="JBFqnCBsd6RMkjVDRZzb",
                        model_id="eleven_multilingual_v2",
                    )

                    audio_bytes = b"".join(chunk for chunk in audio if chunk)

                    st.success("Podcast ready 🎧")
                    st.audio(audio_bytes, format="audio/mp3")

                    st.download_button(
                        "Download",
                        audio_bytes,
                        "podcast.mp3",
                        "audio/mp3",
                    )

                    with st.expander("📄 Script"):
                        st.write(summary)

                else:
                    st.error("Failed to generate")

            except Exception as e:
                st.error(f"Error: {e}")