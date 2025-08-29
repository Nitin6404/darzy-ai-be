#!/usr/bin/env python3
"""
Script to run the Product Search API server
"""

import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """
    Run the FastAPI server
    """
    # Check if Pinecone API key is configured
    if not os.getenv('PINECONE_API_KEY'):
        print("❌ Error: PINECONE_API_KEY not found in environment variables")
        print("Please add your Pinecone API key to the .env file:")
        print("PINECONE_API_KEY=your_api_key_here")
        return
    
    # Check OpenAI API key (optional but recommended)
    openai_configured = bool(os.getenv('OPENAI_API_KEY'))
    
    print("🚀 Starting Product Search API...")
    print("📍 API will be available at: http://localhost:8000")
    print("📖 API documentation at: http://localhost:8000/docs")
    
    print("\n🔍 Available endpoints:")
    print("• POST /search - Vector similarity search")
    print('{"query": "blue patagonia jacket", "top_k": 5}')
    
    if openai_configured:
        print("• POST /agent - LangChain agent with conversational AI")
        print('{"query": "I need a warm jacket for hiking", "conversation_id": "optional"}')
        print("✅ LangChain agent enabled with product search tools")
    else:
        print("• POST /agent - ⚠️  Requires OPENAI_API_KEY in .env file")
        print("Add OPENAI_API_KEY=your_openai_key to enable LangChain agent")
    
    print("\n⏹️  Press Ctrl+C to stop the server")
    
    # Run the server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()