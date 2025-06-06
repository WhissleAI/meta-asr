# applications/session_store.py
import time
from typing import Dict, List, Optional
from config import UserApiKey, USER_API_KEY_TTL_SECONDS, logger

# In-memory store: user_id -> { "api_keys": { "provider": "key" }, "last_updated": timestamp }
_user_sessions: Dict[str, Dict[str, any]] = {}

def init_user_session(user_id: str, api_keys: List[UserApiKey]):
    """Initializes or updates a user's API keys in the session store."""
    logger.info(f"Initializing/updating API keys for user_id: {user_id}")
    user_api_keys_dict = {key.provider: key.key for key in api_keys}
    _user_sessions[user_id] = {
        "api_keys": user_api_keys_dict,
        "last_updated": time.time()
    }

def get_user_api_key(user_id: str, provider: str) -> Optional[str]:
    """Retrieves a specific API key for a user and provider, checking TTL."""
    session = _user_sessions.get(user_id)
    if not session:
        logger.warning(f"No session found for user_id: {user_id}")
        return None

    # Check TTL
    if time.time() - session["last_updated"] > USER_API_KEY_TTL_SECONDS:
        logger.warning(f"API key session expired for user_id: {user_id}. Cleaning up.")
        # Lazy cleanup of expired session
        del _user_sessions[user_id]
        return None

    # Refresh last_updated on access to keep session alive if active
    # session["last_updated"] = time.time() # Optional: uncomment to refresh TTL on each access

    api_key = session["api_keys"].get(provider)
    if not api_key:
        logger.warning(f"No API key found for provider: {provider} for user_id: {user_id}")
        return None
    
    # logger.debug(f"Retrieved API key for user {user_id}, provider {provider}") # Be careful logging keys
    return api_key

def is_user_session_valid(user_id: str) -> bool:
    """Checks if a user session exists and is not expired."""
    session = _user_sessions.get(user_id)
    if not session:
        return False
    if time.time() - session["last_updated"] > USER_API_KEY_TTL_SECONDS:
        return False
    return True

# Optional: Background task for cleaning up very old sessions if lazy cleanup isn't enough
# This would require a separate thread or async task runner.
# def cleanup_expired_sessions():
#     current_time = time.time()
#     expired_users = [
#         user_id for user_id, data in _user_sessions.items() 
#         if current_time - data["last_updated"] > USER_API_KEY_TTL_SECONDS * 2 # Example: cleanup if twice TTL
#     ]
#     for user_id in expired_users:
#         logger.info(f"Background cleanup: Removing expired session for user_id: {user_id}")
#         del _user_sessions[user_id]
