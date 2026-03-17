from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # CRAWL4AI Configuration
    crawl4ai_llm_base_url: str = "http://173.10.88.250:8000/v1"
    
    # VLLM/Qwen Configuration
    vllm_api_key: str = "jaaspire-key"
    vllm_model_name: str = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Environment
    environment: str = "development"
    
    # Logging
    log_level: str = "INFO"

    # Conversation Memory
    memory_db_path: str = "data/conversations.db"
    memory_max_messages: int = 40
    memory_keep_recent: int = 10
    
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="allow")


# Create global settings instance
settings = Settings()
