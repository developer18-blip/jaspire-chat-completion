"""
Conversation memory + Memory Vault using SQLite.

Two layers of memory:
  1. Conversation Memory — stores all messages per conversation_id
     (short-term, per-session). Auto-summarizes when history grows too long.
  2. Memory Vault — stores long-term facts about each user across ALL
     conversations (user interests, preferences, what they asked about,
     personal details they shared). This makes the bot truly "remember" users.
"""

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class ConversationMemory:
    """SQLite-backed conversation memory + long-term memory vault."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or settings.memory_db_path

        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._create_tables()

        logger.info(f"[Memory] Database ready at {self.db_path}")

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversation_messages (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                user_id         TEXT,
                role            TEXT NOT NULL,
                content         TEXT NOT NULL,
                sources         TEXT,
                is_summary      INTEGER DEFAULT 0,
                created_at      REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_conv_id
                ON conversation_messages(conversation_id, created_at);

            CREATE TABLE IF NOT EXISTS memory_vault (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id         TEXT NOT NULL,
                fact_type       TEXT NOT NULL,
                fact            TEXT NOT NULL,
                conversation_id TEXT,
                created_at      REAL NOT NULL,
                updated_at      REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_vault_user
                ON memory_vault(user_id);
        """)
        self._conn.commit()

    def save_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        user_id: Optional[str] = None,
        sources: Optional[list] = None,
    ):
        """Save a single message to the conversation."""
        self._conn.execute(
            """INSERT INTO conversation_messages
               (conversation_id, user_id, role, content, sources, is_summary, created_at)
               VALUES (?, ?, ?, ?, ?, 0, ?)""",
            (
                conversation_id,
                user_id,
                role,
                content,
                json.dumps(sources) if sources else None,
                time.time(),
            ),
        )
        self._conn.commit()

    def get_history(self, conversation_id: str, limit: int = 50) -> list[dict]:
        """
        Return the last `limit` messages for a conversation.
        Summaries come first, then recent messages.
        """
        rows = self._conn.execute(
            """SELECT role, content, sources, is_summary
               FROM conversation_messages
               WHERE conversation_id = ?
               ORDER BY created_at ASC""",
            (conversation_id,),
        ).fetchall()

        # Take the last `limit` messages
        rows = rows[-limit:] if len(rows) > limit else rows

        result = []
        for row in rows:
            msg = {
                "role": row["role"],
                "content": row["content"],
                "is_summary": bool(row["is_summary"]),
            }
            if row["sources"]:
                try:
                    msg["sources"] = json.loads(row["sources"])
                except json.JSONDecodeError:
                    pass
            result.append(msg)

        return result

    def get_conversation_count(self, conversation_id: str) -> int:
        """Return the number of messages in a conversation."""
        row = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM conversation_messages WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        return row["cnt"] if row else 0

    def summarize_if_needed(self, conversation_id: str):
        """
        If conversation has too many messages, summarize old ones
        into a compact summary and delete the originals.
        """
        max_msgs = settings.memory_max_messages
        keep_recent = settings.memory_keep_recent
        count = self.get_conversation_count(conversation_id)

        if count <= max_msgs:
            return

        logger.info(
            f"[Memory] Conversation {conversation_id} has {count} messages, "
            f"summarizing (keep recent {keep_recent})"
        )

        # Get all messages
        rows = self._conn.execute(
            """SELECT id, role, content, is_summary, created_at
               FROM conversation_messages
               WHERE conversation_id = ?
               ORDER BY created_at ASC""",
            (conversation_id,),
        ).fetchall()

        # Split: old messages to summarize, recent to keep
        old_messages = rows[:-keep_recent]
        if not old_messages:
            return

        # Build text to summarize
        convo_text = []
        old_ids = []
        for row in old_messages:
            if row["is_summary"]:
                convo_text.append(f"[Previous Summary]: {row['content']}")
            else:
                convo_text.append(f"{row['role'].upper()}: {row['content']}")
            old_ids.append(row["id"])

        summary = self._generate_summary("\n".join(convo_text))

        if summary:
            # Delete old messages and insert summary
            placeholders = ",".join("?" * len(old_ids))
            self._conn.execute(
                f"DELETE FROM conversation_messages WHERE id IN ({placeholders})",
                old_ids,
            )
            self._conn.execute(
                """INSERT INTO conversation_messages
                   (conversation_id, user_id, role, content, sources, is_summary, created_at)
                   VALUES (?, NULL, 'system', ?, NULL, 1, ?)""",
                (conversation_id, summary, time.time()),
            )
            self._conn.commit()
            logger.info(f"[Memory] Summarized {len(old_ids)} messages into 1 summary")

    def _generate_summary(self, conversation_text: str) -> str:
        """Call the LLM to generate a conversation summary."""
        try:
            payload = {
                "model": settings.vllm_model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a conversation summarizer. Summarize the following "
                            "conversation into 3-5 concise bullet points. Preserve: "
                            "key facts, user preferences, decisions made, and important "
                            "context. Be brief but complete."
                        ),
                    },
                    {"role": "user", "content": conversation_text},
                ],
                "temperature": 0.3,
                "max_tokens": 300,
            }

            resp = httpx.post(
                f"{settings.crawl4ai_llm_base_url}/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {settings.vllm_api_key}",
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()

            if data.get("choices"):
                return data["choices"][0]["message"]["content"]

        except Exception as exc:
            logger.error(f"[Memory] Summarization failed: {exc}")

        return ""

    # ------------------------------------------------------------------
    # Memory Vault — long-term user facts
    # ------------------------------------------------------------------

    def save_vault_facts(
        self,
        user_id: str,
        facts: list[dict],
        conversation_id: Optional[str] = None,
    ):
        """
        Save extracted facts about a user to the vault.
        Each fact: {"type": "interest|preference|personal|topic", "fact": "..."}
        Deduplicates — won't store the same fact twice.
        """
        if not user_id or not facts:
            return

        now = time.time()
        existing = self._get_existing_facts(user_id)

        saved = 0
        for f in facts:
            fact_text = f.get("fact", "").strip()
            fact_type = f.get("type", "topic").strip()
            if not fact_text:
                continue

            # Skip if a very similar fact already exists
            if self._fact_exists(existing, fact_text):
                continue

            self._conn.execute(
                """INSERT INTO memory_vault
                   (user_id, fact_type, fact, conversation_id, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (user_id, fact_type, fact_text, conversation_id, now, now),
            )
            saved += 1

        if saved:
            self._conn.commit()
            logger.info(f"[MemoryVault] Saved {saved} new facts for user {user_id}")

    def _get_existing_facts(self, user_id: str) -> list[str]:
        """Get all existing fact strings for a user."""
        rows = self._conn.execute(
            "SELECT fact FROM memory_vault WHERE user_id = ?",
            (user_id,),
        ).fetchall()
        return [row["fact"].lower() for row in rows]

    def _fact_exists(self, existing: list[str], new_fact: str) -> bool:
        """Check if a similar fact already exists (simple substring match)."""
        new_lower = new_fact.lower()
        for ex in existing:
            # If either contains the other, it's a duplicate
            if new_lower in ex or ex in new_lower:
                return True
        return False

    def get_vault(self, user_id: str, limit: int = 20) -> list[dict]:
        """Get the most recent facts about a user from the vault."""
        if not user_id:
            return []

        rows = self._conn.execute(
            """SELECT fact_type, fact, created_at
               FROM memory_vault
               WHERE user_id = ?
               ORDER BY updated_at DESC
               LIMIT ?""",
            (user_id, limit),
        ).fetchall()

        return [
            {"type": row["fact_type"], "fact": row["fact"]}
            for row in rows
        ]

    def build_vault_context(self, user_id: str) -> str:
        """
        Build a text block of user memories to inject into the system prompt.
        This tells the AI what it knows about this user from past conversations.
        """
        facts = self.get_vault(user_id, limit=20)
        if not facts:
            return ""

        lines = ["\n--- YOUR MEMORY ABOUT THIS USER ---"]
        lines.append("You remember the following about this user from past conversations. "
                     "Use this knowledge to personalize your response:")

        for f in facts:
            label = f["type"].upper()
            lines.append(f"  - [{label}] {f['fact']}")

        lines.append("--- END OF USER MEMORY ---")
        return "\n".join(lines)

    def extract_and_save_facts(
        self,
        user_id: str,
        user_message: str,
        assistant_response: str,
        conversation_id: Optional[str] = None,
    ):
        """
        Call LLM to extract memorable facts from the conversation turn,
        then save them to the vault.
        """
        if not user_id or not user_message:
            return

        try:
            payload = {
                "model": settings.vllm_model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a memory extractor. From the conversation below, "
                            "extract key facts about the USER that are worth remembering "
                            "for future conversations.\n\n"
                            "Extract facts like:\n"
                            "- What topics they are interested in\n"
                            "- Personal details they shared (name, location, job, etc.)\n"
                            "- Preferences they expressed\n"
                            "- What they were looking for or researching\n"
                            "- Problems they are trying to solve\n\n"
                            "Return a JSON array of objects. Each object has:\n"
                            '  "type": one of "interest", "personal", "preference", "topic", "goal"\n'
                            '  "fact": a short sentence describing the fact\n\n'
                            "If there is nothing worth remembering, return an empty array: []\n"
                            "Return ONLY the JSON array, no other text."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"USER said: {user_message}\n\n"
                            f"ASSISTANT replied: {assistant_response[:500]}"
                        ),
                    },
                ],
                "temperature": 0.1,
                "max_tokens": 300,
            }

            resp = httpx.post(
                f"{settings.crawl4ai_llm_base_url}/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {settings.vllm_api_key}",
                    "Content-Type": "application/json",
                },
                timeout=15.0,
            )
            resp.raise_for_status()
            data = resp.json()

            if not data.get("choices"):
                return

            raw = data["choices"][0]["message"]["content"].strip()

            # Parse JSON from the response (handle markdown code blocks)
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            facts = json.loads(raw)
            if isinstance(facts, list) and facts:
                self.save_vault_facts(user_id, facts, conversation_id)

        except (json.JSONDecodeError, Exception) as exc:
            logger.debug(f"[MemoryVault] Fact extraction skipped: {exc}")


# Global instance
memory = ConversationMemory()
