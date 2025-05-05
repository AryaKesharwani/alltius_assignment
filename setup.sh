#!/bin/bash

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p data/processed
mkdir -p data/vectordb

echo "Setup complete! Run 'source venv/bin/activate' to activate the virtual environment." 