import httpx
import logging
from typing import List, Optional
from app.models.chat import SearchResult
from app.config import settings

logger = logging.getLogger(__name__)


class WebSearchService:
    """Service for web search using CRAWL4AI"""
    
    def __init__(self):
        self.base_url = settings.crawl4ai_llm_base_url.rstrip('/')
        self.api_key = settings.vllm_api_key
        self.timeout = 30
    
    async def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """
        Search the web using CRAWL4AI
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of search results
        """
        try:
            logger.info(f"Searching web for query: {query}")
            
            # Prepare the search request for CRAWL4AI
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "query": query,
                "max_results": max_results,
                "include_content": True
            }
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Note: Adjust the endpoint based on CRAWL4AI's actual API
                response = await client.post(
                    f"{self.base_url}/search",
                    json=payload,
                    headers=headers
                )
                response.raise_for_status()
                
            data = response.json()
            results = self._parse_search_results(data.get('results', []))
            
            logger.info(f"Found {len(results)} search results")
            return results
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP Error during web search: {str(e)}")
            # Return empty results on error instead of crashing
            return []
        except Exception as e:
            logger.error(f"Error during web search: {str(e)}")
            return []
    
    def _parse_search_results(self, results: List[dict]) -> List[SearchResult]:
        """Parse raw search results into SearchResult models"""
        parsed_results = []
        
        for result in results:
            try:
                search_result = SearchResult(
                    title=result.get('title', 'No title'),
                    url=result.get('url', ''),
                    content=result.get('content', '')[:1000],  # Limit content length
                    relevance_score=result.get('score', 0.0)
                )
                parsed_results.append(search_result)
            except Exception as e:
                logger.warning(f"Failed to parse search result: {str(e)}")
                continue
        
        return parsed_results


# Global instance
web_search_service = WebSearchService()
