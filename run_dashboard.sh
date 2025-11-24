#!/bin/bash

# Script to run the FinLove Portfolio Dashboard

echo "Starting FinLove Portfolio Dashboard..."
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "Streamlit is not installed. Installing dependencies..."
    pip install -r requirements.txt
fi

# Run the dashboard
streamlit run dashboard.py

