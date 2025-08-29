#!/usr/bin/env python3
"""
Example usage of the ProductToPinecone processor

This script demonstrates how to:
1. Load product data from JSON
2. Process and structure the data
3. Store it in Pinecone vector database
4. Perform similarity searches
"""

import os
from dotenv import load_dotenv
from product_to_pinecone import ProductToPinecone

# Load environment variables
load_dotenv()

def main():
    """
    Example usage of the ProductToPinecone processor
    """
    # Configuration
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    JSON_FILE_PATH = r"C:\Users\nc157\Projects\dont\task\darzy-ai\algolia_product.json"
    INDEX_NAME = "product-vectors-example"
    
    if not PINECONE_API_KEY:
        print("❌ Error: PINECONE_API_KEY not found in environment variables")
        print("Please add your Pinecone API key to the .env file:")
        print("PINECONE_API_KEY=your_api_key_here")
        return
    
    if not os.path.exists(JSON_FILE_PATH):
        print(f"❌ Error: JSON file not found at {JSON_FILE_PATH}")
        return
    
    print("🚀 Starting Product to Pinecone Processing")
    print("="*50)
    
    # Initialize the processor
    processor = ProductToPinecone(PINECONE_API_KEY, INDEX_NAME)
    
    try:
        # Step 1: Process and store products
        # print("📦 Processing and storing products in Pinecone...")
        # processor.process_and_store_products(JSON_FILE_PATH, batch_size=50)
        
        # print("\n✅ Products successfully stored in Pinecone!")

        # Set up index
        processor.setup_index()
        
        # Step 2: Demonstrate search functionality
        print("\n🔍 Testing Search Functionality")
        print("="*30)
        
        # Example searches
        search_queries = [
            "blue patagonia jacket",
            # "men's down sweater",
            # "rain jacket waterproof",
            # "navy blue outerwear"
        ]
        
        for query in search_queries:
            print(f"\n🔎 Searching for: '{query}'")
            results = processor.search_products(query, top_k=3)
            print(results)
            
            if results:
                for i, result in enumerate(results, 1):
                    metadata = result.metadata
                    print(f"  {i}. {metadata.get('title', 'N/A')} - ${metadata.get('price', 'N/A')} (Score: {result.score:.3f})")
                    print(f"     Brand: {metadata.get('brand', 'N/A')} | Color: {metadata.get('color_display', 'N/A')}")
            else:
                print("  No results found")
        
        print("\n🎉 Example completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()