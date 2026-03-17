# 🚀 JASPIRE Chat API - Quick Start Guide

## Project Overview

You now have a **production-ready FastAPI backend** for your Instagram-like application's chatbot feature. It integrates:

- ✅ **CRAWL4AI** - Web search and content crawling
- ✅ **Qwen 3 VL 30B** - Advanced LLM via VLLM
- ✅ **FastAPI** - Modern async Python framework
- ✅ **CORS** - Frontend integration ready

---

## 🎯 System Architecture

```
Your Instagram App (Frontend)
            ↓
    FastAPI Backend (Port 8000)
            ↓
        ├─→ CRAWL4AI Service (Web Search)
        │     └→ http://173.10.88.250:8000/v1
        │
        └─→ VLLM Service (Qwen LLM)
              └→ http://173.10.88.250:8000/v1
```

---

## 📦 Installation & Setup

### **Option 1: Windows (Recommended)**
```bash
# Double-click this file
setup.bat

# Then run the server
python main.py
```

### **Option 2: macOS/Linux**
```bash
bash setup.sh
python main.py
```

### **Option 3: Manual Setup**
```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run server
python main.py
```

---

## 🔌 API Endpoints

### 1. **Send Chat Message** (Main Endpoint)
```
POST /api/chat/message
```
**Request:**
```json
{
  "question": "What's trending on social media?",
  "user_id": "user_123",
  "search_web": true,
  "conversation_history": []
}
```

**Response:**
```json
{
  "answer": "The latest trends include...",
  "sources": [
    {
      "title": "Social Media Trends Report",
      "url": "https://example.com",
      "content": "...",
      "relevance_score": 0.95
    }
  ],
  "search_performed": true,
  "model_used": "Qwen/Qwen3-VL-30B-A3B-Instruct",
  "timestamp": "2024-01-15T10:30:00",
  "user_id": "user_123"
}
```

### 2. **Health Check**
```
GET /api/chat/health
```
Returns: Service status and configuration

### 3. **Get Available Models**
```
GET /api/chat/models
```
Returns: Model info and capabilities

---

## 💻 Frontend Integration Example

### **JavaScript/React**
```javascript
// Install axios or use fetch
async function askBot(question, userId) {
  const response = await fetch('http://localhost:8000/api/chat/message', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question: question,
      user_id: userId,
      search_web: true,
      conversation_history: []
    })
  });
  
  const data = await response.json();
  return data.answer; // Display this in your UI
}

// Usage in a component
const [answer, setAnswer] = React.useState('');
const handleAsk = async (question) => {
  const result = await askBot(question, currentUser.id);
  setAnswer(result);
};
```

### **Python Flask/Django**
```python
import requests

def get_chat_response(question, user_id):
    response = requests.post(
        'http://localhost:8000/api/chat/message',
        json={
            'question': question,
            'user_id': user_id,
            'search_web': True,
            'conversation_history': []
        }
    )
    return response.json()['answer']
```

---

## 📁 Project Structure

```
JASPIRE-API-OPENAI/
├── main.py                    # 🚀 Entry point - Run this!
├── requirements.txt           # Python dependencies
├── .env                       # Configuration (already set)
├── README.md                  # Full documentation
├── example_client.py          # Test script
├── setup.bat/sh              # Setup scripts
│
└── app/
    ├── config.py             # Settings management
    ├── models/
    │   └── chat.py          # Data schemas (Request/Response)
    ├── services/
    │   ├── chat.py          # Main orchestration logic
    │   ├── web_search.py    # CRAWL4AI integration
    │   └── llm.py           # Qwen LLM integration
    └── routes/
        └── chat.py          # API endpoints
```

---

## 🧪 Testing the API

### **Option 1: Use Interactive Docs**
1. Run: `python main.py`
2. Open: `http://localhost:8000/docs`
3. Click "Try it out" on `/api/chat/message`
4. Fill in the fields and execute

### **Option 2: Use Example Client**
```bash
python example_client.py
```

### **Option 3: Use cURL**
```bash
curl -X POST "http://localhost:8000/api/chat/message" \
  -H "Content-Type: application/json" \
  -d '{"question":"Hello!","user_id":"user_123","search_web":true,"conversation_history":[]}'
```

---

## ⚙️ Configuration

Your `.env` file is already configured:
```env
CRAWL4AI_LLM_BASE_URL=http://173.10.88.250:8000/v1      # Web search
VLLM_API_KEY=jaaspire-key                                # API key
VLLM_MODEL_NAME=Qwen/Qwen3-VL-30B-A3B-Instruct          # Qwen model
API_HOST=0.0.0.0                                         # Listen on all IPs
API_PORT=8000                                            # Port
ENVIRONMENT=development                                   # Dev mode
```

**To change settings**, edit `.env` and restart the server.

---

## 🚨 Troubleshooting

### **"Connection refused"**
- Make sure the server is running: `python main.py`
- Check port 8000 is not in use: `lsof -i :8000` (macOS/Linux)

### **"Cannot connect to CRAWL4AI"**
- Verify: `http://173.10.88.250:8000` is reachable from your network
- Update `.env` if the URL changes

### **"Slow responses"**
- Web search takes time (5-10s typical)
- Try with `search_web: false` for faster responses
- Qwen model is large, so responses can take 10-30s

### **"Model not found"**
- Verify `VLLM_MODEL_NAME` in `.env`
- Make sure the model is deployed on the VLLM server

---

## 📊 Expected Response Times

| Operation | Time |
|-----------|------|
| Web Search | 5-15 seconds |
| LLM Response | 5-20 seconds |
| **Total** | **10-35 seconds** |

---

## 🎓 Next Steps

1. **Test locally**: Run `python main.py` and visit `/docs`
2. **Integrate frontend**: Use the API endpoints in your Instagram app
3. **Deploy**: Use a production ASGI server (Gunicorn + Uvicorn)
4. **Scale**: Add load balancing and multi-worker setup

---

## 📚 Full Documentation

See [README.md](README.md) for:
- ✅ Complete API documentation
- ✅ Deployment instructions
- ✅ Production setup
- ✅ Performance optimization
- ✅ Error handling details

---

## 🎉 You're Ready!

Run this command to start your API:
```bash
python main.py
```

Then visit: **http://localhost:8000/docs**

Happy coding! 🚀
