#!/usr/bin/env python
"""
Startup script for Angel One Chatbot
This checks dependencies and starts the Streamlit app.
"""

import os
import sys
import subprocess
import importlib
import time
import platform


def print_status(message, status=""):
    """Print a formatted status message."""
    if status == "OK":
        status_text = "\033[92m[OK]\033[0m"  # Green
    elif status == "ERROR":
        status_text = "\033[91m[ERROR]\033[0m"  # Red
    elif status == "WARNING":
        status_text = "\033[93m[WARNING]\033[0m"  # Yellow
    else:
        status_text = ""
        
    print(f"{message.ljust(60)} {status_text}")

def check_dependency(module_name):
    """Check if a Python module is installed."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

def install_requirements():
    """Install required packages."""
    print_status("Installing requirements...", "")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print_status("Requirements installed successfully.", "OK")
        return True
    except subprocess.CalledProcessError as e:
        print_status(f"Error installing requirements: {e}", "ERROR")
        return False

def check_environment():
    """Check the Python environment."""
    print("\n=== Checking environment ===")
    
    # Check Python version
    python_version = platform.python_version()
    if tuple(map(int, python_version.split('.'))) >= (3, 8):
        print_status(f"Python version: {python_version}", "OK")
    else:
        print_status(f"Python version: {python_version}", "WARNING")
        print("  Python 3.8 or higher is recommended.")
    
    # Check essential dependencies
    essential_deps = [
        "streamlit", 
        "google.generativeai", 
        "pinecone", 
        "torch", 
        "sentence_transformers"
    ]
    
    missing_deps = []
    for dep in essential_deps:
        if check_dependency(dep.split('.')[0]):
            print_status(f"Checking {dep}", "OK")
        else:
            print_status(f"Checking {dep}", "ERROR")
            missing_deps.append(dep)
    
    # Install missing dependencies if needed
    if missing_deps:
        print(f"\nMissing dependencies: {', '.join(missing_deps)}")
        if input("Do you want to install missing dependencies? (y/n): ").lower() == 'y':
            install_requirements()
    
    # Check data directories
    processed_data_path = "data/processed/combined_documents.txt"
    if os.path.exists(processed_data_path):
        print_status("Processed data exists", "OK")
    else:
        print_status("Processed data not found", "WARNING")
        print("  You will need to run data processing first.")
    
    # Check API keys
    env_content = open(".env").read() if os.path.exists(".env") else ""
    
    if "GOOGLE_API_KEY" in env_content:
        print_status("Google Gemini API key found in .env file", "OK")
    else:
        print_status("Google Gemini API key not found", "WARNING")
        print("  You can still use basic retrieval, but Gemini AI features will be disabled.")
    
    if "PINECONE_API_KEY" in env_content:
        print_status("Pinecone API key found in .env file", "OK")
    else:
        print_status("Pinecone API key not found", "ERROR")
        print("  Pinecone API key is required. Please add it to your .env file.")
        print("  You can get a free API key from https://www.pinecone.io/")
        
    print("\nEnvironment check complete.\n")

def start_app():
    """Start the Streamlit app."""
    print("Starting Angel One Support Chatbot...\n")
    time.sleep(1)
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"\nError starting application: {e}")
        print("\nTry running manually with: streamlit run app.py")

if __name__ == "__main__":
    print("\n=== Angel One Support Chatbot ===")
    check_environment()
    
    if input("Start the application? (y/n): ").lower() in ('y', 'yes', ''):
        start_app()
    else:
        print("\nExiting setup. You can start manually with: streamlit run app.py") 