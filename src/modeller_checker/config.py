"""Configuration loader for modeller-checker workflow."""

import os
from pathlib import Path
from typing import Optional, Dict, Any

import yaml
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml. If None, looks for config.yaml in repo root.
    
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Look for config.yaml in repo root
        repo_root = Path(__file__).resolve().parents[2]
        config_path = repo_root / "config.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)
    
    # Extract modeller_checker config or use legacy format
    if 'modeller_checker' in full_config:
        config = full_config['modeller_checker']
    else:
        # Legacy format support
        config = full_config
    
    # Expand environment variables in API keys
    for agent in ['modeller', 'checker']:
        if agent in config and 'api_key' in config[agent]:
            api_key = config[agent]['api_key']
            if api_key and api_key.startswith('${') and api_key.endswith('}'):
                env_var = api_key[2:-1]
                config[agent]['api_key'] = os.getenv(env_var)
    
    return config


def create_llm_from_config(config: Dict[str, Any], defaults: Dict[str, Any] = None) -> BaseChatModel:
    """
    Create LLM instance from configuration.
    
    Args:
        config: LLM configuration dict with provider, model, etc.
        defaults: Optional defaults dict (e.g., llm_defaults from config)
    
    Returns:
        Configured LLM instance
    
    Raises:
        ValueError: If provider is not supported
    """
    defaults = defaults or {}
    provider = config.get("provider", "ollama").lower()
    default_max_tokens = defaults.get("default_max_tokens", 2048)
    default_temperature = defaults.get("default_temperature", 0.5)
    
    if provider == "ollama":
        return ChatOllama(
            model=config.get("model", "qwen3"),
            temperature=config.get("temperature", default_temperature),
            base_url=config.get("base_url", "http://127.0.0.1:11434"),
        )
    
    elif provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "langchain-openai not installed. Install with: pip install langchain-openai"
            )
        
        return ChatOpenAI(
            model=config.get("model", "gpt-4"),
            temperature=config.get("temperature", default_temperature),
            api_key=config.get("api_key"),
            max_tokens=config.get("max_tokens"),
        )
    
    elif provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(
                "langchain-anthropic not installed. Install with: pip install langchain-anthropic"
            )
        
        return ChatAnthropic(
            model=config.get("model", "claude-3-sonnet-20240229"),
            temperature=config.get("temperature", default_temperature),
            api_key=config.get("api_key"),
            max_tokens=config.get("max_tokens", default_max_tokens),
        )
    
    elif provider == "azure":
        try:
            from langchain_openai import AzureChatOpenAI
        except ImportError:
            raise ImportError(
                "langchain-openai not installed. Install with: pip install langchain-openai"
            )
        
        return AzureChatOpenAI(
            model=config.get("model", "gpt-4"),
            temperature=config.get("temperature", default_temperature),
            api_key=config.get("api_key"),
            azure_endpoint=config.get("azure_endpoint"),
            api_version=config.get("api_version", "2024-02-15-preview"),
        )
    
    else:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            f"Supported: ollama, openai, anthropic, azure"
        )


def create_llms_from_config(
    config_path: Optional[str] = None
) -> tuple[BaseChatModel, BaseChatModel]:
    """
    Create modeller and checker LLMs from config file.
    
    Args:
        config_path: Path to config.yaml
    
    Returns:
        (modeller_llm, checker_llm) tuple
    """
    config = load_config(config_path)
    llm_defaults = config.get("llm_defaults", {})
    
    modeller_llm = create_llm_from_config(config["modeller"], defaults=llm_defaults)
    checker_llm = create_llm_from_config(config["checker"], defaults=llm_defaults)
    
    return modeller_llm, checker_llm
