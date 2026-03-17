# JASPIRE Chat API

OpenAI-compatible FastAPI chatbot backend for an Instagram-like application.

## What This Service Provides

- OpenAI-style endpoint: `POST /v1/chat/completions`
- Same request payload style as OpenAI Chat Completions
- Streaming (`stream=true`) and non-streaming responses
- LangChain-based web search path (`search_web=true`)
- Automatic fallback search mode if upstream tool-calling is unavailable

## One-Command Install And Run

### Windows

```bash
setup.bat
```

### Linux/macOS

```bash
bash setup.sh
```

Both scripts do all steps automatically:
1. Create virtual environment (if missing)
2. Install package (`pip install -e .`)
3. Start API and run startup checks (`jaspire auto --with-search`)

## Manual Package Mode

```bash
python -m venv venv
venv\Scripts\activate
pip install -e .
jaspire auto --with-search
```

## CLI Commands

- `jaspire init` - Create `.env` if missing
- `jaspire run` - Start the FastAPI server
- `jaspire doctor --with-search` - Run health checks
- `jaspire auto --with-search` - Start server + run checks automatically

## API Base URL

`http://localhost:8000/v1`

## Main Endpoint

```http
POST /v1/chat/completions
Content-Type: application/json
```

### Request Example

```json
{
  "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
  "messages": [
    { "role": "user", "content": "Latest AI news today" }
  ],
  "stream": false,
  "search_web": true,
  "temperature": 0.7
}
```

### Non-Streaming Response Shape

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1710000000,
  "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  }
}
```

## Health And Model Endpoints

- `GET /v1/health`
- `GET /v1/models`

## Backend Integration Pattern

Recommended flow:
1. Frontend calls your application backend.
2. Your backend forwards payload to this service (`/v1/chat/completions`).
3. Backend returns/streams response to frontend.

This keeps internal AI infrastructure private and secure.

## Example Backend Proxy (Node/Express)

```javascript
app.post('/api/bot/message', async (req, res) => {
  const upstream = await fetch('http://localhost:8000/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...req.body, stream: false })
  });
  const data = await upstream.json();
  res.json(data);
});
```

## Notes

- If port 8000 is already in use, stop the previous process or change port.
- For production, add authentication, rate limiting, structured logging, and monitoring.
