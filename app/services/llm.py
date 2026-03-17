import httpx
import logging
from typing import List, Optional, Dict, Any
from app.models.chat import ChatMessage
from app.config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """Service for interacting with Qwen LLM via VLLM"""
    
    def __init__(self):
        self.base_url = settings.crawl4ai_llm_base_url.rstrip('/')
        self.api_key = settings.vllm_api_key
        self.model = settings.vllm_model_name
        self.timeout = 60
    
    async def generate_response(
        self,
        question: str,
        conversation_history: Optional[List[ChatMessage]] = None,
        context: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> str:
        """
        Generate a response using Qwen LLM
        
        Args:
            question: User's question
            conversation_history: Previous messages in conversation
            context: Additional context (e.g., from web search)
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            
        Returns:
            Generated response text
        """
        try:
            logger.info(f"Generating response for question: {question[:50]}...")
            
            # Build conversation messages
            messages = self._build_messages(question, conversation_history, context)
            
            # Prepare request headers
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Prepare request payload for OpenAI-compatible API
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": 0.95,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            }
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers
                )
                response.raise_for_status()
            
            data = response.json()
            
            # Extract the response text
            if data.get('choices') and len(data['choices']) > 0:
                answer = data['choices'][0].get('message', {}).get('content', '')
                logger.info("Response generated successfully")
                return answer
            else:
                logger.warning("Unexpected response format from LLM")
                return "I couldn't generate a proper response. Please try again."
                
        except httpx.HTTPError as e:
            logger.error(f"HTTP Error during LLM call: {str(e)}")
            return f"Error communicating with the model: {str(e)}"
        except Exception as e:
            logger.error(f"Error during response generation: {str(e)}")
            return f"An error occurred: {str(e)}"
    
    def _build_messages(
        self,
        question: str,
        conversation_history: Optional[List[ChatMessage]] = None,
        context: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Build message list for LLM API"""
        messages = []
        
        # Add system message with context
        system_message = "You are a helpful AI assistant for an Instagram-like social application. "
        if context:
            system_message += f"Use the following information to answer the question: {context}"
        
        messages.append({"role": "system", "content": system_message})
        
        # Add conversation history
        if conversation_history:
            for msg in conversation_history[-5:]:  # Keep last 5 messages for context
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        # Add current question
        messages.append({"role": "user", "content": question})
        
        return messages


# Global instance
llm_service = LLMService()
