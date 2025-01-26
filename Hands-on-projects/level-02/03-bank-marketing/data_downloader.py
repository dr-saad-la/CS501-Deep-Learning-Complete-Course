"""
    Bank marketing dataset downloader
    
"""

import os
import sys
from pathlib import Path
import subprocess
import zipfile
import io
import shutil
from urllib.parse import urlparse, unquote

from datetime import datetime
from typing import Tuple, Dict, List

import requests
from tqdm import tqdm

def run_tree_command(path: str, show_hidden: bool = True, show_perms: bool = True) -> None:
    """
    Run the 'tree' command with specified options to display directory structure.
    
    Args:
        path (str): Path to the directory to analyze
        show_hidden (bool): If True, show hidden files (default: True)
        show_perms (bool): If True, show file permissions (default: True)
        
    Raises:
        FileNotFoundError: If tree command is not installed
        subprocess.CalledProcessError: If tree command fails
        ValueError: If path is invalid
    """
    if not path:
        raise ValueError("Path cannot be empty")
        
    if not os.path.isdir(path):
        raise ValueError(f"Invalid directory path: {path}")
    
    # Build command with options
    cmd = ['tree']
    
    if show_hidden:
        cmd.append('-a')  # Show hidden files
    
    if show_perms:
        cmd.append('-l')  # Show permissions
        
    cmd.append(path)
    
    try:
        output = subprocess.check_output(cmd, text=True, stderr=subprocess.PIPE)
        print(output)
    except FileNotFoundError:
        print("The 'tree' command is not installed on your system. Install it and try again.")
        print("On Ubuntu/Debian: sudo apt-get install tree")
        print("On macOS: brew install tree")
        print("On Windows: Install via chocolatey: choco install tree")
    except subprocess.CalledProcessError as e:
        print(f"Tree command failed with error: {e.stderr}")
    except Exception as e:
        print(f"Unexpected error occurred: {str(e)}")

def get_filename_from_url(url):
    """Extract filename from URL, handling encoded characters"""
    parsed_url = urlparse(url)
    filename = unquote(os.path.basename(parsed_url.path))
    return filename if filename else 'downloaded_file.zip'


def download_with_progress(url, save_path):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    
    with open(save_path, 'wb') as f:
        with tqdm(total=total_size, unit='iB', unit_scale=True, desc="Downloading") as pbar:
            for data in response.iter_content(block_size):
                size = f.write(data)
                pbar.update(size)
    
    return save_path


def extract_nested_zip(zip_path, extract_path, remove_after=True):
    """
    Extract nested zip files recursively
    
    Args:
        zip_path: Path to the zip file
        extract_path: Path where to extract contents
        remove_after: Whether to remove zip files after extraction
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Check for dangerous paths (path traversal attack prevention)
            for name in zip_ref.namelist():
                if name.startswith('/') or '..' in name:
                    raise ValueError(f"Potentially dangerous path in zip: {name}")
            
            # Extract all contents with progress bar
            total_files = len(zip_ref.namelist())
            with tqdm(total=total_files, desc="Extracting") as pbar:
                for member in zip_ref.namelist():
                    zip_ref.extract(member, extract_path)
                    pbar.update(1)
                    
                    # Handle nested zip files
                    if member.lower().endswith('.zip'):
                        nested_path = os.path.join(extract_path, member)
                        nested_extract_path = os.path.join(
                            extract_path, 
                            os.path.splitext(member)[0]
                        )
                        
                        # Recursively extract nested zip
                        if os.path.exists(nested_path):
                            extract_nested_zip(nested_path, nested_extract_path)
                            if remove_after:
                                os.remove(nested_path)
        
        return True
    
    except Exception as e:
        print(f"Error extracting {zip_path}: {str(e)}")
        return False

    
def download_zip_data(url, save_path='data', keep_zip=False):
    """
    Download and extract data from a given url, handling nested zip files
    
    Args:
        url: URL of the zip file to download
        save_path: Local path where to save the extracted files
        keep_zip: Whether to keep the downloaded zip file
    
    Returns:
        tuple: (success_status, extracted_path)
    """
    try:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        filename = get_filename_from_url(url)
        temp_zip = save_path.joinpath(filename)
        
        print(f"Downloading from {url}...")
        download_with_progress(url, temp_zip)
        
        print("\nExtracting files...")
        extraction_success = extract_nested_zip(temp_zip, save_path)
        
        # Clean up
        if not keep_zip and temp_zip.exists():
            temp_zip.unlink()
        
        if extraction_success:
            print("\nExtracted contents:")
            # List extracted files with size
            for item in save_path.rglob('*'):
                if item.is_file():
                    size = item.stat().st_size
                    rel_path = item.relative_to(save_path)
                    print(f"{rel_path} ({humanize_size(size)})")
            
            return True, save_path
        else:
            return False, None
            
    except requests.exceptions.RequestException as e:
        print(f"Download error: {str(e)}")
        return False, None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return False, None

def humanize_size(size):
    """Convert size in bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def remove_hidden_directories(directory: str) -> None:
    """
    Recursively remove all hidden directories (directories starting with '.' and __MACOSX)
    from the specified directory and its subdirectories.
    
    Args:
        directory (str): Path to the directory to clean
        
    Raises:
        OSError: If there's an error accessing or removing directories
        ValueError: If the directory path is invalid
    """
    if not directory:
        raise ValueError("Directory path cannot be empty")
        
    if not os.path.isdir(directory):
        raise ValueError(f"Invalid directory path: {directory}")
    
    try:
        for root, dirs, _ in os.walk(directory, topdown=False):
            for dirname in dirs:
                if dirname.startswith('.') or dirname == '__MACOSX':
                    dir_path = os.path.join(root, dirname)
                    if os.path.isdir(dir_path):
                        try:
                            os.rmdir(dir_path)  # Try to remove if empty
                            print(f"Removed empty hidden directory: {dir_path}")
                        except OSError:
                            # If directory not empty, use shutil.rmtree()
                            import shutil
                            shutil.rmtree(dir_path)
                            print(f"Removed hidden directory and its contents: {dir_path}")
                    
    except OSError as e:
        print(f"Failed to remove hidden directories: {str(e)}")
        raise
    except Exception as e:
        print(f"Unexpected error occurred: {str(e)}")
        raise


def remove_macosx_artifacts(data_path: str) -> None:
    """
    Recursively search and remove all macOS-specific '__MACOSX' directories if they exist.
    
    Args:
        data_path (str): Path to the parent directory to search for macOS artifacts
        
    Raises:
        OSError: If there's an error during directory removal
        TypeError: If data_path is not a string
        ValueError: If data_path is empty or invalid
    """
    if not isinstance(data_path, str):
        raise TypeError("data_path must be a string")
        
    if not data_path:
        raise ValueError("data_path cannot be empty")
        
    if not os.path.isdir(data_path):
        raise ValueError(f"Invalid directory path: {data_path}")
        
    try:
        # Walk through the directory tree bottom-up
        for root, dirs, _ in os.walk(data_path, topdown=False):
            if '__MACOSX' in dirs:
                macosx_dir = os.path.join(root, '__MACOSX')
                if os.path.isdir(macosx_dir):
                    shutil.rmtree(macosx_dir)
                    print(f"Successfully removed macOS artifact directory: {macosx_dir}")
                    
    except OSError as e:
        logger.error(f"Failed to remove macOS artifact directory: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error occurred: {str(e)}")
        raise     
        
def remove_hidden_files(directory: str) -> None:
    """
    Recursively remove all hidden files (files starting with '.') from the specified directory
    and its subdirectories.
    
    Args:
        directory (str): Path to the directory to clean
        
    Raises:
        OSError: If there's an error accessing or removing files
        ValueError: If the directory path is invalid
    """
    if not directory:
        raise ValueError("Directory path cannot be empty")
        
    if not os.path.isdir(directory):
        raise ValueError(f"Invalid directory path: {directory}")
        
    try:
        # Walk through directory and all subdirectories
        for root, dirs, files in os.walk(directory, topdown=False):
            # Remove hidden files
            for filename in files:
                if filename.startswith('.'):
                    file_path = os.path.join(root, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        print(f"Removed hidden file: {file_path}")
                    
    except OSError as e:
        print(f"Failed to remove hidden files: {str(e)}")
        raise
    except Exception as e:
        print(f"Unexpected error occurred: {str(e)}")
        raise
