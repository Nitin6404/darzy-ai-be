#!/usr/bin/env python3
"""
Example usage of the LangChain Agent API endpoint

This script demonstrates how to interact with the /agent endpoint
for conversational product recommendations.
"""

import requests
import json
from typing import Optional

class AgentClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.conversation_id = None
    
    def chat(self, query: str, conversation_id: Optional[str] = None) -> dict:
        """
        Send a query to the LangChain agent
        
        Args:
            query: The user's question or request
            conversation_id: Optional conversation ID for context
            
        Returns:
            Agent response dictionary
        """
        url = f"{self.base_url}/agent"
        payload = {
            "query": query,
            "conversation_id": conversation_id or self.conversation_id
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            
            # Update conversation ID if provided in response
            if result.get('conversation_id'):
                self.conversation_id = result['conversation_id']
            
            return result
        except requests.exceptions.RequestException as e:
            return {"error": f"Request failed: {str(e)}"}
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON response: {str(e)}"}

def main():
    """
    Example usage of the LangChain Agent
    """
    print("🤖 LangChain Agent Example")
    print("=" * 30)
    
    # Initialize client
    client = AgentClient()
    
    # Example conversations
    queries = [
        "I'm looking for a warm jacket for winter hiking",
        "What blue jackets do you have from Patagonia?",
        "Can you recommend something waterproof?",
        "What's the price range for these jackets?",
        "Tell me about the materials used in outdoor jackets"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n🔍 Query {i}: {query}")
        print("-" * 50)
        
        response = client.chat(query)
        
        if "error" in response:
            print(f"❌ Error: {response['error']}")
        else:
            print(f"🤖 Agent: {response.get('response', 'No response')}")
            if response.get('conversation_id'):
                print(f"💬 Conversation ID: {response['conversation_id']}")
        
        print()
    
    # Interactive mode
    print("\n" + "=" * 50)
    print("🎯 Interactive Mode (type 'quit' to exit)")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            response = client.chat(user_input)
            
            if "error" in response:
                print(f"❌ Error: {response['error']}")
            else:
                print(f"🤖 Agent: {response.get('response', 'No response')}")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
    
    print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()