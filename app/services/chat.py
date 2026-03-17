import logging
from typing import Optional, List
from app.models.chat import ChatRequest, ChatResponse, ChatMessage, SearchResult
from app.services.web_search import web_search_service
from app.services.llm import llm_service
from app.config import settings

logger = logging.getLogger(__name__)


class ChatService:
    """Main chat service orchestrating web search and LLM"""
    
    async def process_chat(self, chat_request: ChatRequest) -> ChatResponse:
        """
        Process a chat request by searching the web and generating a response
        
        Args:
            chat_request: The chat request with question and context
            
        Returns:
            ChatResponse with generated answer and sources
        """
        sources = []
        search_performed = False
        
        try:
            # Step 1: Search the web if requested
            context = None
            if chat_request.search_web:
                logger.info(f"Searching web for question: {chat_request.question}")
                sources = await web_search_service.search(chat_request.question)
                search_performed = True
                
                # Build context from search results
                if sources:
                    context = self._build_context_from_sources(sources)
                    logger.info(f"Built context from {len(sources)} sources")
            
            # Step 2: Generate response using LLM
            logger.info("Generating LLM response")
            answer = await llm_service.generate_response(
                question=chat_request.question,
                conversation_history=chat_request.conversation_history,
                context=context,
                temperature=0.7,
                max_tokens=512
            )
            
            # Step 3: Build and return response
            response = ChatResponse(
                answer=answer,
                sources=sources,
                search_performed=search_performed,
                model_used=chat_request.model or settings.vllm_model_name,
                user_id=chat_request.user_id
            )
            
            logger.info(f"Chat processing complete for user {chat_request.user_id}")
            return response
            
        except Exception as e:
            logger.error(f"Error processing chat: {str(e)}")
            return ChatResponse(
                answer=f"An error occurred while processing your request: {str(e)}",
                sources=[],
                search_performed=search_performed,
                model_used=chat_request.model or settings.vllm_model_name,
                user_id=chat_request.user_id
            )
    
    def _build_context_from_sources(self, sources: List[SearchResult]) -> str:
        """Build context string from search results"""
        context_parts = []
        
        for i, source in enumerate(sources[:3], 1):  # Use top 3 sources
            context_parts.append(f"Source {i}: {source.title}\n{source.content}")
        
        return "\n\n".join(context_parts)


# Global instance
chat_service = ChatService()
