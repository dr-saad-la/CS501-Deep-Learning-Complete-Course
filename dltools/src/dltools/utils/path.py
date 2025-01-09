import os
import shutil
import logging
from typing import Union
from pathlib import Path

def cleanup_directory(directory_path: Union[str, Path], create_after: bool = False) -> bool:
    """
    Safely removes a directory and all its contents.
    
    This utility function handles the deletion of directories and their contents,
    with proper error handling and logging. It can optionally recreate the
    directory after deletion.
    
    Args:
        directory_path: Path to the directory to be deleted. Can be either a string
            or a Path object.
        create_after: If True, creates an empty directory at the same location
            after deletion. Defaults to False.
    
    Returns:
        bool: True if the operation was successful, False otherwise.
    
    Raises:
        TypeError: If directory_path is neither a string nor a Path object.
        PermissionError: If the process lacks necessary permissions.
    
    Example:
        >>> cleanup_directory("/path/to/directory")
        True
        >>> cleanup_directory("/path/to/directory", create_after=True)
        True
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Convert string path to Path object if necessary
        directory = Path(directory_path) if isinstance(directory_path, str) else directory_path
        
        if not isinstance(directory, Path):
            raise TypeError("directory_path must be a string or Path object")
            
        if not directory.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return False
            
        if not directory.is_dir():
            logger.error(f"Path exists but is not a directory: {directory}")
            return False
            
        # Remove directory and its contents
        shutil.rmtree(directory)
        logger.info(f"Successfully removed directory: {directory}")
        
        # Optionally recreate the directory
        if create_after:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created new empty directory: {directory}")
            
        return True
        
    except PermissionError as e:
        logger.error(f"Permission denied while accessing {directory}: {str(e)}")
        raise
        
    except Exception as e:
        logger.error(f"An error occurred while cleaning up {directory}: {str(e)}")
        return False