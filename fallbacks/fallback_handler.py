"""Helper module for managing fallback implementations."""
import os
import sys
from typing import Any, Dict, Optional, Callable, Tuple, Type, TypeVar

T = TypeVar('T')

def import_with_fallback(
    main_module_path: str,
    main_class_name: str,
    fallback_module_path: str,
    fallback_class_name: str,
    logger_func: Optional[Callable[[str, str], None]] = None
) -> Tuple[Type[Any], bool]:
    """
    Attempts to import a class from the main module, falling back to a fallback implementation if needed.
    
    Args:
        main_module_path: Import path for the main implementation (e.g., 'mission_manager')
        main_class_name: Name of the main class to import
        fallback_module_path: Import path for the fallback implementation
        fallback_class_name: Name of the fallback class
        logger_func: Optional logging function for status messages
    
    Returns:
        Tuple of (class_reference, is_fallback)
    """
    log = logger_func if logger_func else lambda lvl, msg: print(f"[{lvl}] {msg}")
    
    try:
        module = __import__(main_module_path, fromlist=[main_class_name])
        class_ref = getattr(module, main_class_name)
        log("DEBUG", f"Successfully imported {main_class_name} from {main_module_path}")
        return class_ref, False
    except (ImportError, AttributeError) as e:
        log("WARNING", f"Failed to import {main_class_name} from {main_module_path}: {e}")
        try:
            fallback_module = __import__(fallback_module_path, fromlist=[fallback_class_name])
            fallback_class = getattr(fallback_module, fallback_class_name)
            log("INFO", f"Using fallback implementation {fallback_class_name} from {fallback_module_path}")
            return fallback_class, True
        except (ImportError, AttributeError) as e_fb:
            log("ERROR", f"Failed to import fallback {fallback_class_name} from {fallback_module_path}: {e_fb}")
            raise ImportError(f"Neither main nor fallback implementation could be imported for {main_class_name}")

def create_instance_with_fallback(
    main_module_path: str,
    main_class_name: str,
    fallback_module_path: str,
    fallback_class_name: str,
    constructor_args: Dict[str, Any],
    logger_func: Optional[Callable[[str, str], None]] = None
) -> Tuple[Any, bool]:
    """
    Creates an instance of a class, falling back to a fallback implementation if needed.
    
    Args:
        main_module_path: Import path for the main implementation
        main_class_name: Name of the main class to import
        fallback_module_path: Import path for the fallback implementation
        fallback_class_name: Name of the fallback class
        constructor_args: Dictionary of arguments to pass to the constructor
        logger_func: Optional logging function for status messages
    
    Returns:
        Tuple of (instance, is_fallback)
    """
    try:
        class_ref, is_fallback = import_with_fallback(
            main_module_path, main_class_name,
            fallback_module_path, fallback_class_name,
            logger_func
        )
        instance = class_ref(**constructor_args)
        return instance, is_fallback
    except Exception as e:
        if logger_func:
            logger_func("ERROR", f"Failed to create instance of {main_class_name}: {e}")
        raise
