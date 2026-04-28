import asyncpg
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from loguru import logger


class ConversationDB:
    """Stores chat history in Postgres. Each row is one message tagged with
    'human message: ' or 'aimessage: '."""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        self.pool = await asyncpg.create_pool(dsn=self.database_url)
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id SERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    message TEXT NOT NULL
                )
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages (session_id)"
            )
        logger.info("Connected to Postgres; messages table ready")

    async def close(self) -> None:
        if self.pool is not None:
            await self.pool.close()
            logger.info("Postgres pool closed")

    async def save_human_message(self, session_id: str, content: str) -> None:
        await self._insert(session_id, f"human message: {content}")

    async def save_ai_message(self, session_id: str, content: str) -> None:
        await self._insert(session_id, f"aimessage: {content}")

    async def get_last_messages(
        self, session_id: str, limit: int = 10
    ) -> list[BaseMessage]:
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT message FROM messages
                    WHERE session_id = $1
                    ORDER BY id DESC
                    LIMIT $2
                    """,
                    session_id,
                    limit,
                )
        except Exception as e:
            logger.error(f"Failed to read messages for session {session_id}: {e}")
            return []

        # DB gave us newest-first; flip so the LLM sees them chronologically.
        messages: list[BaseMessage] = []
        for row in reversed(rows):
            text = row["message"]
            if text.startswith("human message: "):
                messages.append(HumanMessage(content=text.removeprefix("human message: ")))
            elif text.startswith("aimessage: "):
                messages.append(AIMessage(content=text.removeprefix("aimessage: ")))
        return messages

    async def _insert(self, session_id: str, message: str) -> None:
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO messages (session_id, message) VALUES ($1, $2)",
                    session_id,
                    message,
                )
        except Exception as e:
            # Persistence failure shouldn't break the chat; log and move on.
            logger.error(f"Failed to save message for session {session_id}: {e}")
