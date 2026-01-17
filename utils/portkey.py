from portkey_ai import Portkey
import os

def get_portkey_client(provider: str = None, model: str = None):
    """
    Creates a Portkey client using the Model Catalog.
    
    Args:
        provider: Optional provider name (for backward compatibility)
        model: Optional model slug to use
    
    Returns:
        Portkey client instance
    """
    api_key = os.getenv("PORTKEY_API_KEY")
    
    if not api_key:
        raise ValueError("PORTKEY_API_KEY environment variable not set")
    
    # Create base Portkey client with API key
    client = Portkey(
        api_key=api_key,
        base_url="https://api.portkey.ai/v1",
    )
    
    return client


def get_portkey_client_with_config(config: dict):
    """
    Creates a Portkey client with a specific configuration.
    
    Args:
        config: Configuration dict with model routing/fallback rules
    
    Returns:
        Portkey client instance
    """
    api_key = os.getenv("PORTKEY_API_KEY")
    
    if not api_key:
        raise ValueError("PORTKEY_API_KEY environment variable not set")
    
    return Portkey(
        api_key=api_key,
        config=config
    )