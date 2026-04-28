import uuid

import requests
import streamlit as st

BACKEND_CHAT_URL = "http://127.0.0.1:8000/chat"


def stream_chat_response(session_id: str, prompt: str, backend_url: str):
    payload = {"session_id": session_id, "payload": prompt}

    with requests.post(
        backend_url,
        json=payload,
        stream=True,
        timeout=(10, 300),
    ) as response:
        response.raise_for_status()

        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                yield chunk


st.set_page_config(page_title="Streaming RAG Chat", page_icon="💬", layout="centered")
st.title("Streaming RAG Chat")
st.caption("Streamlit client for your FastAPI `/chat` endpoint")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Connection")
    backend_url = st.text_input("Backend chat URL", value=BACKEND_CHAT_URL)
    session_id = st.text_input("Session ID", value=st.session_state.session_id)

    if session_id != st.session_state.session_id:
        st.session_state.session_id = session_id

    if st.button("Generate New Session ID"):
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Type your message")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            response_stream = stream_chat_response(
                session_id=st.session_state.session_id,
                prompt=prompt,
                backend_url=backend_url,
            )
            full_response = st.write_stream(response_stream)
        except requests.HTTPError as exc:
            full_response = (
                f"Backend returned HTTP {exc.response.status_code}: "
                f"{exc.response.text}"
            )
            st.error(full_response)
        except requests.RequestException as exc:
            full_response = f"Request failed: {exc}"
            st.error(full_response)

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )
