"""
Example usage of the JASPIRE Chat API client
"""
import httpx
import asyncio
import json


class JASPIREChatClient:
    """Client for interacting with JASPIRE Chat API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.client = None
    
    async def __aenter__(self):
        self.client = httpx.AsyncClient()
        return self
    
    async def __aexit__(self, *args):
        if self.client:
            await self.client.aclose()
    
    async def check_health(self):
        """Check API health"""
        response = await self.client.get(f"{self.base_url}/api/chat/health")
        response.raise_for_status()
        return response.json()
    
    async def send_message(
        self,
        question: str,
        user_id: str,
        search_web: bool = True,
        conversation_history: list = None
    ):
        """Send a message and get a response"""
        payload = {
            "question": question,
            "user_id": user_id,
            "search_web": search_web,
            "conversation_history": conversation_history or []
        }
        
        response = await self.client.post(
            f"{self.base_url}/api/chat/message",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    async def get_models(self):
        """Get available models"""
        response = await self.client.get(f"{self.base_url}/api/chat/models")
        response.raise_for_status()
        return response.json()


async def main():
    """Example usage"""
    async with JASPIREChatClient() as client:
        # Check health
        print("🏥 Checking API health...\n")
        try:
            health = await client.check_health()
            print(f"✅ API Status: {health['status']}")
            print(f"   CRAWL4AI: {health['crawl4ai_url']}")
            print(f"   Model: {health['vllm_model']}\n")
        except Exception as e:
            print(f"❌ Health check failed: {e}\n")
            return
        
        # Get available models
        print("🤖 Available Models:\n")
        try:
            models = await client.get_models()
            print(json.dumps(models, indent=2))
            print()
        except Exception as e:
            print(f"❌ Failed to get models: {e}\n")
        
        # Send a message
        print("💬 Sending message...\n")
        try:
            response = await client.send_message(
                question="What are the latest developments in artificial intelligence?",
                user_id="user_123",
                search_web=True
            )
            
            print(f"📝 Question: What are the latest developments in AI?\n")
            print(f"🤖 Answer:\n{response['answer']}\n")
            
            if response.get('sources'):
                print("📚 Sources:")
                for i, source in enumerate(response['sources'], 1):
                    print(f"  {i}. {source['title']}")
                    print(f"     URL: {source['url']}")
                    if source.get('relevance_score'):
                        print(f"     Relevance: {source['relevance_score']:.2%}\n")
            
            print(f"⏱️  Processed with model: {response['model_used']}")
            
        except httpx.ConnectError:
            print("❌ Could not connect to API. Make sure the server is running:")
            print("   python main.py")
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
