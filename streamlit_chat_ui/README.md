# Streamlit Chat UI

This folder contains a simple Streamlit chatbot client for the FastAPI `/chat`
endpoint.

## Run

```bash
uv run streamlit run streamlit_chat_ui/app.py
```

## Notes

- Update the placeholder `BACKEND_CHAT_URL` in `app.py`, or change it from the
  Streamlit sidebar at runtime.
- The app sends this JSON payload to the backend:

```json
{
  "session_id": "your-session-id",
  "payload": "your-user-message"
}
```

- The response is rendered incrementally as soon as the backend starts sending
  streamed chunks.
