# JASPIRE Chat API - Manager Handoff Package

## What You Have

A production-ready OpenAI-compatible chat API service with:
- ✅ Web-search intelligence (LangChain powered)
- ✅ Streaming support (same as ChatGPT)
- ✅ Automatic fallback if tool-calling unavailable
- ✅ One-command setup and run

## Installation (Choose One)

### Option 1: One-Click Setup (Recommended)

**Windows:**
```
Double-click: setup.bat
```

**Linux/macOS:**
```
bash setup.sh
```

This does everything:
1. Creates virtual environment
2. Installs dependencies
3. Starts API server
4. Runs health checks
5. Shows you the running API

### Option 2: Manual Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows: .\venv\Scripts\activate
pip install -e .
jaspire auto --with-search
```

## What You Get

API running at: `http://localhost:8000/v1`

Main endpoint:
```
POST /v1/chat/completions
```

Same payload format as OpenAI:
```json
{
  "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
  "messages": [
    { "role": "user", "content": "Latest AI news" }
  ],
  "stream": false,
  "search_web": true,
  "temperature": 0.7
}
```

## Ready to Integrate

Your app backend should forward requests to:
```
POST http://localhost:8000/v1/chat/completions
```

Example (Node.js):
```javascript
const response = await fetch('http://localhost:8000/v1/chat/completions', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(userPayload)
});
```

## Verification

After startup, you'll see:
```
[OK] /health
[OK] /models  
[OK] /chat/completions without search
[OK] /chat/completions with search
[RESULT] All checks passed
```

If anything fails, check logs and ensure:
- Python 3.10+ is installed
- Port 8000 is free
- Network access to http://173.10.88.250:8000/v1 (upstream VLLM+CRAWL4AI)

## Next Steps for Your Team

1. **Dev**: Integrate this service into your app backend
2. **QA**: Test with your frontend messaging flow
3. **Prod**: Add auth, rate limiting, logging

## Support

- Run `jaspire --help` for all CLI options
- Check `README.md` for detailed API documentation
- Review `pyproject.toml` for dependency versions
- Look at `app/cli.py` to understand automation commands

## Files Included

```
JASPIRE-API-OPENAI/
├── setup.bat / setup.sh    ← Run one of these
├── pyproject.toml          ← Package metadata
├── requirements.txt        ← Dependencies
├── .env                    ← Pre-configured credentials
├── README.md               ← Full docs
├── main.py                 ← FastAPI entry point
├── example_client.py       ← Test client code
└── app/                    ← Application code
    ├── cli.py              ← Automation commands
    ├── config.py           ← Settings
    ├── models/             ← Data schemas
    ├── services/           ← Business logic
    └── routes/             ← API endpoints
```

---

**Ready to go?** Run setup and start integrating. 🚀
