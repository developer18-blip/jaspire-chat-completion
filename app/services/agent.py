"""
Chat service with web search + conversation memory.

Flow:
  1. Load conversation history from memory (if conversation_id provided).
  2. AUTO-DETECT if web search is needed (or respect explicit override).
  3. Build prompt: system + history + search context + current message.
  4. Call Qwen LLM via vLLM.
  5. Save user message + assistant response to memory.
  6. Return answer with sources.
"""

import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import AsyncGenerator, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.config import settings
from app.services.tools import run_web_search
from app.services.memory import memory
from app.services.recommendations import get_recommendations, build_recommendation_context

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Smart search detection — Two-layer approach:
#   Layer 1: FORCE SKIP patterns (greetings, math, code, memory recall)
#   Layer 2: NEED WEB signals (if query has real-time/factual keywords → search)
#   Default: If no signal either way, DON'T search (answer from AI knowledge)
# ---------------------------------------------------------------------------

# Layer 1: ALWAYS skip search for these (no matter what)
_FORCE_SKIP_PATTERNS = [
    # Code writing / debugging
    r"\b(write|create|generate|build|make)\b.{0,20}\b(code|program|script|function|class|api|app|endpoint)\b",
    r"\b(fix|debug|refactor|optimize)\b.{0,20}\b(code|bug|error|function|script)\b",
    r"```",  # Contains code block
    # Memory recall
    r"\bwhat did (i|we|you) (say|ask|tell|mention)\b",
    r"\bdo you remember\b",
    r"\bwhat('s| is) my name\b",
    r"\btumhe\s*yaad\s*hai\b",
    r"\bmaine\s*kya\s*(kaha|pucha|bataya)\b",
    # Translation
    r"^translate\b",
    r"^(summarize|paraphrase|rewrite)\b.*[:\"']",
    # Bot identity / casual questions (never need web search)
    r"\bwho\s*are\s*you\b",
    r"\bwhat\s*is\s*your\s*name\b",
    r"\bwhat('s| is)\s*your\s*name\b",
    r"\bwho\s*made\s*you\b",
    r"\bwho\s*created\s*you\b",
    r"\btumhara\s*naam\b",
    r"\btum\s*kaun\s*ho\b",
    r"\baap\s*kaun\b",
    r"\btell\s*me\s*about\s*yourself\b",
    r"\bintroduce\s*yourself\b",
    # Capability questions (bot should answer from identity, not search)
    r"\bcan\s*you\s*(do|perform|use|run)\s*(web\s*)?search\b",
    r"\bdo\s*you\s*(have|support)\s*(web\s*)?search\b",
    r"\bcan\s*you\s*search\s*(the\s*)?(web|internet|online)\b",
    r"\bwhat\s*can\s*you\s*do\b",
    r"\bwhat\s*are\s*your\s*(features|capabilities|abilities)\b",
]

_FORCE_SKIP_RE = [re.compile(p, re.IGNORECASE) for p in _FORCE_SKIP_PATTERNS]

# Identity question patterns — used to reinforce persona in the user message
_IDENTITY_PATTERNS = [
    r"\bwho\s*are\s*you\b",
    r"\bwhat\s*is\s*your\s*name\b",
    r"\bwhat('s| is)\s*your\s*name\b",
    r"\bwho\s*made\s*you\b",
    r"\bwho\s*created\s*you\b",
    r"\btumhara\s*naam\b",
    r"\btum\s*kaun\s*ho\b",
    r"\baap\s*kaun\b",
    r"\btell\s*me\s*about\s*yourself\b",
    r"\bintroduce\s*yourself\b",
]
_IDENTITY_RE = [re.compile(p, re.IGNORECASE) for p in _IDENTITY_PATTERNS]


def _is_identity_question(query: str) -> bool:
    """Check if the query is asking about the bot's identity."""
    for pattern in _IDENTITY_RE:
        if pattern.search(query):
            return True
    return False


# Capability question patterns
_CAPABILITY_PATTERNS = [
    r"\bcan\s*you\s*(do|perform|use|run)\s*(web\s*)?search\b",
    r"\bdo\s*you\s*(have|support)\s*(web\s*)?search\b",
    r"\bcan\s*you\s*search\s*(the\s*)?(web|internet|online)\b",
    r"\bwhat\s*can\s*you\s*do\b",
    r"\bwhat\s*are\s*your\s*(features|capabilities|abilities)\b",
]
_CAPABILITY_RE = [re.compile(p, re.IGNORECASE) for p in _CAPABILITY_PATTERNS]


def _is_capability_question(query: str) -> bool:
    """Check if the query is asking about bot capabilities."""
    for pattern in _CAPABILITY_RE:
        if pattern.search(query):
            return True
    return False

# Layer 2: Keywords that SIGNAL the query needs real-time web data
_WEB_SIGNAL_KEYWORDS = [
    # Time-sensitive / real-time
    "latest", "current", "today", "tonight", "yesterday", "this week",
    "this month", "this year", "right now", "breaking", "update",
    "recent", "new", "2024", "2025", "2026", "2027",
    # News & events
    "news", "war", "attack", "election", "president", "government",
    "crisis", "killed", "dead", "bomb", "missile", "conflict",
    "earthquake", "flood", "hurricane", "storm", "disaster",
    "protest", "riot", "coup", "sanctions", "ceasefire",
    # Prices / markets / finance
    "price", "cost", "stock", "market", "bitcoin", "crypto",
    "gold", "silver", "oil", "dollar", "rupee", "exchange rate",
    "inflation", "gdp", "economy", "salary", "worth",
    # Sports / entertainment
    "score", "match", "game", "tournament", "champion", "winner",
    "ipl", "world cup", "oscar", "grammy", "movie", "release date",
    "box office", "trailer", "season",
    # People / companies (looking up specific info)
    "ceo", "founder", "net worth", "biography", "age of",
    # Weather
    "weather", "temperature", "forecast", "rain", "snow",
    # Lookup / research
    "how to", "where to", "best", "top", "review", "compare",
    "vs", "versus", "recommendation", "suggest", "find",
    "near me", "in seattle", "in new york", "in india",
    "where is", "where can", "who is", "who was", "when did",
    "when is", "when was",
    # Hindi / Urdu signals for real-time queries
    "aaj", "abhi", "taza", "khabar", "samachar", "kimat",
    "kab", "kahan", "kaun", "konsa", "naya", "nayi",
]

# Common words that are NOT search-worthy (greetings, casual, filler)
_CASUAL_WORDS = {
    # English greetings & filler
    "hi", "hii", "hiii", "hello", "hey", "yo", "sup", "hiya", "howdy",
    "ok", "okay", "yes", "no", "yeah", "yep", "nope", "sure", "hmm",
    "thanks", "thank", "you", "bye", "goodbye", "good", "morning",
    "evening", "night", "afternoon", "how", "are", "is", "it", "do",
    "doing", "fine", "great", "cool", "nice", "awesome", "wow", "lol",
    "haha", "hehe", "please", "sorry", "well", "oh", "ah", "um",
    "right", "alright", "of", "course", "a", "an", "the", "and", "or",
    "me", "my", "i", "am", "was", "tell", "can", "could", "would",
    "will", "what", "that", "this", "its", "to", "too", "so", "but",
    "not", "just", "very", "really", "much", "also", "about", "know",
    # Hindi / Urdu casual
    "or", "aur", "kaise", "ho", "hai", "hain", "kya", "hal", "theek",
    "thik", "accha", "acha", "ji", "haan", "nahi", "bhai", "yaar",
    "dost", "batao", "bolo", "sunao", "bas", "chalo", "chal", "sahi",
    "bohot", "bahut", "badhiya", "mast", "shukriya", "namaste",
    "kaisa", "haal", "raha", "rahi", "rahe", "hu", "hoon", "tum",
    "aap", "main", "mein", "bro", "sir", "mam", "mujhe", "tujhe",
}


def needs_web_search(query: str) -> bool:
    """
    Smart two-layer search detection:

    1. FORCE SKIP: Code, memory recall, translation → never search
    2. WEB SIGNALS: If query contains real-time keywords (news, price, latest,
       who is, where, weather, etc.) → search the web
    3. CASUAL CHECK: If all words are casual/greeting words → skip
    4. DEFAULT: Don't search (let AI answer from knowledge)

    Examples:
      "tell me 2+2"           → no web signal, casual → SKIP
      "how are you"           → all casual words → SKIP
      "or kaise ho"           → all casual words → SKIP
      "what is python"        → no web signal → SKIP (AI knows this)
      "latest iran news"      → "latest" + "news" = web signal → SEARCH
      "gold price today"      → "price" + "today" = web signal → SEARCH
      "who is elon musk"      → "who is" = web signal → SEARCH
      "best surgeon seattle"  → "best" + "in seattle" = web signal → SEARCH
    """
    if not query or not query.strip():
        return False

    q = query.strip()
    q_lower = q.lower()
    words = q.split()

    # Single word → always skip
    if len(words) == 1:
        logger.info(f"[AutoSearch] SKIP — single word: {q[:60]}")
        return False

    # Layer 1: Force-skip patterns (code, memory, translation)
    for pattern in _FORCE_SKIP_RE:
        if pattern.search(q):
            logger.info(f"[AutoSearch] SKIP — force-skip pattern: {q[:60]}")
            return False

    # Layer 2: Check for web-signal keywords
    for keyword in _WEB_SIGNAL_KEYWORDS:
        if keyword in q_lower:
            logger.info(f"[AutoSearch] YES — web signal '{keyword}' found: {q[:60]}")
            return True

    # Layer 3: If it's a short message with all casual words → skip
    clean_words = [re.sub(r'[!?.,:;+\-*/=()]+', '', w).lower() for w in words]
    clean_words = [w for w in clean_words if w]  # remove empty after stripping

    if all(w in _CASUAL_WORDS for w in clean_words):
        logger.info(f"[AutoSearch] SKIP — all casual/common words: {q[:60]}")
        return False

    # Layer 4: Check if it looks like a question about a real-world entity/event
    # Questions starting with "who/what/where/when/why/how" + proper nouns → search
    if re.match(r"^(who|where|when|why)\s", q_lower):
        logger.info(f"[AutoSearch] YES — factual question: {q[:60]}")
        return True

    # Default: DON'T search — let the AI answer from its knowledge
    # This catches: "tell me 2+2", "explain gravity", "what is python",
    # "write a poem", "define machine learning", etc.
    logger.info(f"[AutoSearch] SKIP — no web signal, using AI knowledge: {q[:60]}")
    return False

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
def _get_system_prompt() -> str:
    """Build system prompt with current date."""
    today = datetime.now().strftime("%B %d, %Y")
    return f"""\
You are Jaasi, a friendly AI buddy inside the Jaaspire app. Today's date is {today}.

YOUR PERSONALITY:
- You are warm, friendly, and conversational — like chatting with a fun, helpful friend.
- Use a casual, approachable tone with occasional emojis when appropriate.
- For greetings like "hi", "hello", "who are you" — reply in a friendly, personal way. Introduce yourself as Jaasi, the AI buddy in the Jaaspire app.
- Think of yourself as the user's go-to chat pal for fun convos, jokes, helpful info, and a little help with their social media adventures.
- Keep responses concise but warm and engaging.

YOUR CAPABILITIES:
- You CAN do web search! You have a built-in web search feature that lets you fetch up-to-date information with sources.
- You can look up real-time information from the internet, like:
  * Latest news and current events
  * Weather updates
  * Stock prices / sports scores
  * Local places (restaurants, shops, etc.)
  * Any real-time or factual information
- When someone asks "can you do web search" or "what can you do", proudly tell them about your capabilities!
- NEVER say "I can't do web searches" or "I don't have real-time access" — because you DO have these capabilities.

WHEN WEB SEARCH RESULTS ARE PROVIDED:
1. Extract the ACTUAL information from the search results and present it directly.
2. For restaurant/place queries: give a numbered list of specific place NAMES, what they're known for, and location. Do NOT just list article titles or source links.
3. For news queries: give specific facts, names, numbers — not just article headlines.
4. NEVER say "Based on search results from [date]", "According to the most up-to-date search results", or mention when the search was done.
5. NEVER say "I don't have real-time access" — you DO have real-time access.
6. Answer naturally as if you already know this information — like a knowledgeable friend.
7. Use emojis and bullet points to organize the info nicely.
8. Put source links at the very end in a small "Sources:" section, NOT next to each item.
9. At the end, offer to help with more details.

EXAMPLE FORMAT for restaurant/place queries:
"Here are some awesome brunch spots in Seattle! 🍳
1. **Portage Bay Cafe** — Known for their farm-to-table breakfast, amazing French toast toppings bar
2. **Biscuit Bitch** — Famous for their fluffy biscuit sandwiches, always a line out the door
3. **Toulouse Petit** — Great Cajun-Creole brunch, try their beignets!
..."

GENERAL RULES:
1. If NO search results are provided, answer using your general knowledge in a friendly way.
2. If conversation history is provided, use it to maintain context.
3. If USER MEMORY is provided, use it to personalize your response naturally (e.g., "Since you were interested in X..." or "As you mentioned before...").
4. Reply in the same language the user used.
5. Keep responses clear, helpful, and fun!
"""


@dataclass
class AgentResponse:
    """Response from the agent with all metadata."""
    answer: str = ""
    sources: list[dict] = field(default_factory=list)
    conversation_id: str = ""
    search_performed: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _build_llm(temperature: float, max_tokens: int, streaming: bool):
    return ChatOpenAI(
        base_url=settings.crawl4ai_llm_base_url,
        api_key=settings.vllm_api_key,
        model=settings.vllm_model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming,
        model_kwargs={"stream_options": {"include_usage": True}} if streaming else {},
    )


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------

def _extract_user_query(openai_messages: list) -> str:
    """Get the latest user message text."""
    for msg in reversed(openai_messages):
        role = msg.role if hasattr(msg, "role") else msg.get("role", "")
        if role != "user":
            continue
        content = msg.content if hasattr(msg, "content") else msg.get("content", "")
        if isinstance(content, list):
            return " ".join(
                item["text"] if isinstance(item, dict) and item.get("type") == "text"
                else str(item)
                for item in content
            )
        return content or ""
    return ""


def _flatten_content(content) -> str:
    """Flatten string or list content to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return ""


def _build_messages(
    openai_messages: list,
    history: list[dict],
    web_context: str = "",
    recommendation_context: str = "",
    vault_context: str = "",
) -> list:
    """
    Build LangChain messages:
      1. System prompt
      2. Conversation history from memory
      3. Current request messages (with search results injected INTO the user message)

    KEY DESIGN: Search results are injected into the user's message, NOT the system
    prompt. This forces the model to read them as part of the question, making it
    much harder to ignore them and fall back to training data.
    """
    messages = []

    # 1. System prompt (clean — no search results here)
    system_content = _get_system_prompt()
    if vault_context:
        system_content += vault_context
    if recommendation_context:
        system_content += recommendation_context
    messages.append(SystemMessage(content=system_content))

    # 2. Conversation history from memory
    for msg in history:
        if msg.get("is_summary"):
            messages.append(SystemMessage(
                content=f"Previous conversation summary:\n{msg['content']}"
            ))
        elif msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))

    # 3. Current request messages (skip system — we already have ours)
    #    Inject search results into the LAST user message
    user_messages_content = []
    other_messages = []

    for msg in openai_messages:
        role = msg.role if hasattr(msg, "role") else msg.get("role", "user")
        content = msg.content if hasattr(msg, "content") else msg.get("content", "")
        content = _flatten_content(content) or ""

        if role == "user":
            user_messages_content.append(content)
        elif role == "assistant":
            other_messages.append(("assistant", content))

    # Add any assistant messages from the request
    for role, content in other_messages:
        messages.append(AIMessage(content=content))

    # Build the final user message with search results embedded
    final_user_text = user_messages_content[-1] if user_messages_content else ""

    if web_context and final_user_text:
        augmented_user_msg = (
            f"{final_user_text}\n\n"
            f"---\n"
            f"WEB SEARCH RESULTS:\n\n"
            f"{web_context}\n\n"
            f"---\n"
            f"INSTRUCTION: Extract the actual information from the results above and answer directly. "
            f"Give specific names, facts, and details — NOT just article titles or source links. "
            f"For places/restaurants: list specific place names with what they're known for. "
            f"Do NOT mention search dates. Answer naturally like you already know this. "
            f"Put source links in a small section at the very end only."
        )
        messages.append(HumanMessage(content=augmented_user_msg))
    elif final_user_text:
        # For identity/greeting questions, reinforce persona directly
        if _is_identity_question(final_user_text):
            reinforced = (
                f"{final_user_text}\n\n"
                f"REMEMBER: Your name is Jaasi. You are a friendly AI buddy inside the Jaaspire app. "
                f"Introduce yourself warmly with emojis. You help users with fun convos, jokes, "
                f"helpful info, and social media adventures on Jaaspire. "
                f"Do NOT say 'I am an AI assistant'. Say 'I'm Jaasi' instead."
            )
            messages.append(HumanMessage(content=reinforced))
        elif _is_capability_question(final_user_text):
            reinforced = (
                f"{final_user_text}\n\n"
                f"REMEMBER: You are Jaasi and you CAN do web search! You have a built-in web search "
                f"feature that lets you fetch up-to-date information from the internet. "
                f"You can look up: latest news, weather updates, stock prices, sports scores, "
                f"local places, and any real-time information. "
                f"Answer proudly and confidently with emojis. List your capabilities with bullet points. "
                f"NEVER say you cannot search the web — because you CAN."
            )
            messages.append(HumanMessage(content=reinforced))
        else:
            messages.append(HumanMessage(content=final_user_text))

    return messages


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def get_agent_response(
    openai_messages: list,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    with_search: Optional[bool] = None,
    conversation_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> AgentResponse:
    """Return the full answer with sources and memory (non-streaming)."""

    # Generate conversation_id if not provided
    conv_id = conversation_id or f"conv-{uuid.uuid4().hex[:12]}"

    # Step 1: Load conversation history
    history = []
    if conversation_id:
        history = memory.get_history(conversation_id, limit=30)
        logger.info(f"[Agent] Loaded {len(history)} history messages for {conv_id}")

    # Step 2: Smart web search decision
    query = _extract_user_query(openai_messages)
    sources = []
    web_context = ""
    search_performed = False

    # Auto-detect if not explicitly set
    should_search = with_search if with_search is not None else needs_web_search(query)

    if should_search and query:
        search_result = run_web_search(query)
        web_context = search_result.context
        sources = search_result.sources
        search_performed = bool(web_context)

    # Step 3: Load Memory Vault (long-term user facts)
    vault_context = ""
    if user_id:
        vault_context = memory.build_vault_context(user_id)
        if vault_context:
            logger.info(f"[Agent] Loaded memory vault for user {user_id}")

    # Step 4: Check for promoted recommendations
    rec_context = ""
    if query:
        recs = get_recommendations(query)
        if recs:
            rec_context = build_recommendation_context(recs)
            logger.info(f"[Agent] Injecting {len(recs)} recommendation(s)")

    # Step 5: Build messages
    lc_messages = _build_messages(openai_messages, history, web_context, rec_context, vault_context)

    logger.info(
        f"[Agent] Non-stream | conv={conv_id} msgs={len(lc_messages)} "
        f"history={len(history)} search={search_performed}"
    )

    # Step 6: Call LLM
    llm = _build_llm(temperature, max_tokens, streaming=False)
    reply = await llm.ainvoke(lc_messages)
    answer = _flatten_content(getattr(reply, "content", "")) or ""

    # Extract actual token usage from Qwen/vLLM
    usage_meta = getattr(reply, "response_metadata", {})
    token_usage = (
        usage_meta.get("token_usage")
        or usage_meta.get("usage")
        or {}
    )
    prompt_tokens = token_usage.get("prompt_tokens", 0)
    completion_tokens = token_usage.get("completion_tokens", 0)
    total_tokens = token_usage.get("total_tokens", 0)

    # Step 7: Save to memory
    user_query = _extract_user_query(openai_messages)
    if user_query:
        memory.save_message(conv_id, "user", user_query, user_id=user_id)
    if answer:
        memory.save_message(conv_id, "assistant", answer, user_id=user_id, sources=sources)

    # Step 8: Extract facts to Memory Vault (background, non-blocking)
    if user_id and user_query:
        try:
            memory.extract_and_save_facts(user_id, user_query, answer, conv_id)
        except Exception as exc:
            logger.debug(f"[Agent] Vault extraction skipped: {exc}")

    # Step 9: Summarize if needed
    try:
        memory.summarize_if_needed(conv_id)
    except Exception as exc:
        logger.warning(f"[Agent] Summarization failed: {exc}")

    return AgentResponse(
        answer=answer,
        sources=sources,
        conversation_id=conv_id,
        search_performed=search_performed,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


async def stream_agent_response(
    openai_messages: list,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    with_search: Optional[bool] = None,
    conversation_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> tuple[AsyncGenerator[str, None], AgentResponse]:
    """
    Returns (token_generator, response_metadata).
    The caller can stream tokens from the generator,
    and use response_metadata for sources/conversation_id.
    """
    conv_id = conversation_id or f"conv-{uuid.uuid4().hex[:12]}"

    # Step 1: Load history
    history = []
    if conversation_id:
        history = memory.get_history(conversation_id, limit=30)
        logger.info(f"[Agent] Loaded {len(history)} history messages for {conv_id}")

    # Step 2: Smart web search decision
    query = _extract_user_query(openai_messages)
    sources = []
    web_context = ""
    search_performed = False

    should_search = with_search if with_search is not None else needs_web_search(query)

    if should_search and query:
        search_result = run_web_search(query)
        web_context = search_result.context
        sources = search_result.sources
        search_performed = bool(web_context)

    # Step 3: Load Memory Vault
    vault_context = ""
    if user_id:
        vault_context = memory.build_vault_context(user_id)
        if vault_context:
            logger.info(f"[Agent] Loaded memory vault for user {user_id}")

    # Step 4: Check for promoted recommendations
    rec_context = ""
    if query:
        recs = get_recommendations(query)
        if recs:
            rec_context = build_recommendation_context(recs)
            logger.info(f"[Agent] Injecting {len(recs)} recommendation(s)")

    # Step 5: Build messages
    lc_messages = _build_messages(openai_messages, history, web_context, rec_context, vault_context)

    logger.info(
        f"[Agent] Stream | conv={conv_id} msgs={len(lc_messages)} "
        f"history={len(history)} search={search_performed}"
    )

    # Prepare metadata (answer will be filled after streaming)
    meta = AgentResponse(
        sources=sources,
        conversation_id=conv_id,
        search_performed=search_performed,
    )

    async def _generate():
        llm = _build_llm(temperature, max_tokens, streaming=True)
        full_answer = []

        async for chunk in llm.astream(lc_messages):
            text = _flatten_content(getattr(chunk, "content", None))
            if text:
                full_answer.append(text)
                yield text

        # After streaming is done, calculate tokens and save to memory
        answer = "".join(full_answer)
        meta.answer = answer

        # Estimate token usage for streaming (vLLM doesn't return usage in stream mode)
        # Use ~4 chars per token approximation for prompt
        prompt_text = " ".join(
            m.content if hasattr(m, 'content') and isinstance(m.content, str) else ''
            for m in lc_messages
        )
        meta.prompt_tokens = max(len(prompt_text) // 4, 1)
        meta.completion_tokens = max(len(answer) // 4, 1) if answer else 0
        meta.total_tokens = meta.prompt_tokens + meta.completion_tokens

        user_query = _extract_user_query(openai_messages)
        if user_query:
            memory.save_message(conv_id, "user", user_query, user_id=user_id)
        if answer:
            memory.save_message(conv_id, "assistant", answer, user_id=user_id, sources=sources)

        # Extract facts to Memory Vault
        if user_id and user_query:
            try:
                memory.extract_and_save_facts(user_id, user_query, answer, conv_id)
            except Exception as exc:
                logger.debug(f"[Agent] Vault extraction skipped: {exc}")

        try:
            memory.summarize_if_needed(conv_id)
        except Exception as exc:
            logger.warning(f"[Agent] Summarization failed: {exc}")

    return _generate(), meta
