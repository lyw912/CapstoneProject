"""
Retry mechanism utility module
Provides generic network request retry functionality to enhance system robustness
"""

import time
from functools import wraps
from typing import Callable, Any
import requests
from loguru import logger

# Configure logging
class RetryConfig:
    """Retry configuration class"""
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        max_delay: float = 60.0,
        retry_on_exceptions: tuple = None
    ):
        """
        Initialize retry configuration
        
        Args:
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds
            backoff_factor: Backoff factor (delay doubles with each retry)
            max_delay: Maximum delay in seconds
            retry_on_exceptions: Tuple of exception types to retry on
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay
        
        # Default exception types to retry on
        if retry_on_exceptions is None:
            self.retry_on_exceptions = (
                requests.exceptions.RequestException,
                requests.exceptions.ConnectionError,
                requests.exceptions.HTTPError,
                requests.exceptions.Timeout,
                requests.exceptions.TooManyRedirects,
                ConnectionError,
                TimeoutError,
                Exception  # OpenAI和其他API可能抛出的一般异常
            )
        else:
            self.retry_on_exceptions = retry_on_exceptions

# Default configuration
DEFAULT_RETRY_CONFIG = RetryConfig()

def with_retry(config: RetryConfig = None):
    """
    Retry decorator
    
    Args:
        config: Retry configuration, uses default if not provided
    
    Returns:
        Decorator function
    """
    if config is None:
        config = DEFAULT_RETRY_CONFIG
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(config.max_retries + 1):  # +1 because first attempt doesn't count as retry
                try:
                    result = func(*args, **kwargs)
                    if attempt > 0:
                        logger.info(f"Function {func.__name__} succeeded after {attempt + 1} attempts")
                    return result
                    
                except config.retry_on_exceptions as e:
                    last_exception = e
                    
                    if attempt == config.max_retries:
                        # Last attempt also failed
                        logger.error(f"Function {func.__name__} failed after {config.max_retries + 1} attempts")
                        logger.error(f"Final error: {str(e)}")
                        raise e
                    
                    # Calculate delay time
                    delay = min(
                        config.initial_delay * (config.backoff_factor ** attempt),
                        config.max_delay
                    )
                    
                    logger.warning(f"Function {func.__name__} failed on attempt {attempt + 1}: {str(e)}")
                    logger.info(f"Will retry in {delay:.1f} seconds (attempt {attempt + 2})...")
                    
                    time.sleep(delay)
                
                except Exception as e:
                    # Exceptions not in retry list, raise immediately
                    logger.error(f"Function {func.__name__} encountered non-retryable exception: {str(e)}")
                    raise e
            
            # This should not be reached, but as a safety net
            if last_exception:
                raise last_exception
            
        return wrapper
    return decorator

def retry_on_network_error(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0
):
    """
    Retry decorator specifically for network errors (simplified version)
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Backoff factor
    
    Returns:
        Decorator function
    """
    config = RetryConfig(
        max_retries=max_retries,
        initial_delay=initial_delay,
        backoff_factor=backoff_factor
    )
    return with_retry(config)

class RetryableError(Exception):
    """Custom retryable exception"""
    pass

def with_graceful_retry(config: RetryConfig = None, default_return=None):
    """
    Graceful retry decorator - for non-critical API calls
    Returns default value instead of raising exception on failure to ensure system continuity
    
    Args:
        config: Retry configuration, uses default if not provided
        default_return: Default value to return when all retries fail
    
    Returns:
        Decorator function
    """
    if config is None:
        config = SEARCH_API_RETRY_CONFIG
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(config.max_retries + 1):  # +1 because first attempt doesn't count as retry
                try:
                    result = func(*args, **kwargs)
                    if attempt > 0:
                        logger.info(f"Non-critical API {func.__name__} succeeded after {attempt + 1} attempts")
                    return result
                    
                except config.retry_on_exceptions as e:
                    last_exception = e
                    
                    if attempt == config.max_retries:
                        # Last attempt also failed, return default instead of raising
                        logger.warning(f"Non-critical API {func.__name__} failed after {config.max_retries + 1} attempts")
                        logger.warning(f"Final error: {str(e)}")
                        logger.info(f"Returning default value to ensure system continuity: {default_return}")
                        return default_return
                    
                    # Calculate delay time
                    delay = min(
                        config.initial_delay * (config.backoff_factor ** attempt),
                        config.max_delay
                    )
                    
                    logger.warning(f"Non-critical API {func.__name__} failed on attempt {attempt + 1}: {str(e)}")
                    logger.info(f"Will retry in {delay:.1f} seconds (attempt {attempt + 2})...")
                    
                    time.sleep(delay)
                
                except Exception as e:
                    # Exceptions not in retry list, return default
                    logger.warning(f"Non-critical API {func.__name__} encountered non-retryable exception: {str(e)}")
                    logger.info(f"Returning default value to ensure system continuity: {default_return}")
                    return default_return
            
            # This should not be reached, but as a safety net
            return default_return
            
        return wrapper
    return decorator

def make_retryable_request(
    request_func: Callable,
    *args,
    max_retries: int = 5,
    **kwargs
) -> Any:
    """
    Directly execute a retryable request (without using decorator)
    
    Args:
        request_func: Request function to execute
        *args: Positional arguments to pass to request function
        max_retries: Maximum number of retry attempts
        **kwargs: Keyword arguments to pass to request function
    
    Returns:
        Return value of request function
    """
    config = RetryConfig(max_retries=max_retries)
    
    @with_retry(config)
    def _execute():
        return request_func(*args, **kwargs)
    
    return _execute()

# Predefined common retry configurations
LLM_SHORT_RETRY_CONFIG = RetryConfig(
    max_retries=2,
    initial_delay=3.0,
    backoff_factor=2.0,
    max_delay=15.0,
)

LLM_LONG_RETRY_CONFIG = RetryConfig(
    max_retries=1,
    initial_delay=10.0,
    backoff_factor=2.0,
    max_delay=60.0,
)

# Backward-compatible alias for long streaming calls
LLM_RETRY_CONFIG = LLM_LONG_RETRY_CONFIG

SEARCH_API_RETRY_CONFIG = RetryConfig(
    max_retries=5,        # Increase to 5 retries
    initial_delay=2.0,    # Increase initial delay
    backoff_factor=1.6,   # Adjust backoff factor
    max_delay=25.0        # Increase maximum delay
)

DB_RETRY_CONFIG = RetryConfig(
    max_retries=5,        # Increase to 5 retries
    initial_delay=1.0,    # Keep shorter delay for database retries
    backoff_factor=1.5,
    max_delay=10.0
)
