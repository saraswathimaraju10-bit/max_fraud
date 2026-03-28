"""
Fraud Detection App - Streamlit Cloud Entry Point
This file allows Streamlit Cloud to deploy the fraud detection app
"""
import sys
import os

# Add backend directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

# Import and run the Streamlit app
from app import *
