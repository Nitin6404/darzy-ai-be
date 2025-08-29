import json
import os
from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import hashlib
from datetime import datetime

class ProductToPinecone:
    def __init__(self, pinecone_api_key: str, index_name: str = "product-vectors"):
        """
        Initialize the ProductToPinecone processor
        
        Args:
            pinecone_api_key: Your Pinecone API key
            index_name: Name of the Pinecone index to use
        """
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index_name = index_name
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight embedding model
        self.index = None
        
    def setup_index(self, dimension: int = 384, metric: str = "cosine"):
        """
        Create or connect to Pinecone index
        
        Args:
            dimension: Vector dimension (384 for all-MiniLM-L6-v2)
            metric: Distance metric for similarity search
        """
        try:
            # Check if index exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                print(f"Creating new index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=dimension,
                    metric=metric,
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
            else:
                print(f"Using existing index: {self.index_name}")
                
            self.index = self.pc.Index(self.index_name)
            print(f"Connected to index: {self.index_name}")
            
        except Exception as e:
            print(f"Error setting up index: {e}")
            raise
    
    def load_product_data(self, json_file_path: str) -> Dict[str, Any]:
        """
        Load product data from JSON file
        
        Args:
            json_file_path: Path to the Algolia product JSON file
            
        Returns:
            Loaded JSON data
        """
        try:
            with open(json_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            print(f"Loaded product data from: {json_file_path}")
            return data
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            raise
    
    def structure_product_data(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """
        Structure product data for better organization
        
        Args:
            product: Raw product data from Algolia
            
        Returns:
            Structured product data
        """
        # Extract key information
        structured = {
            'id': str(product.get('id', '')),
            'title': product.get('title', ''),
            'vendor': product.get('vendor', ''),
            'product_type': product.get('product_type', ''),
            'handle': product.get('handle', ''),
            'price': product.get('price', 0),
            'compare_at_price': product.get('compare_at_price', 0),
            'price_ratio': product.get('price_ratio', 0),
            'tags': product.get('tags', []),
            'collections': product.get('collections', []),
            'options': product.get('options', {}),
            'created_at': product.get('created_at', ''),
            'updated_at': product.get('updated_at', ''),
        }
        
        # Extract metadata if available
        meta = product.get('meta', {})
        if meta:
            # Trove facets
            trove_facets = meta.get('trove-facets', {})
            structured['category'] = trove_facets.get('category', [])
            structured['color'] = trove_facets.get('color', [])
            structured['department'] = trove_facets.get('department', [])
            structured['gender'] = trove_facets.get('gender', [])
            structured['size'] = trove_facets.get('size', [])
            structured['condition'] = trove_facets.get('condition', [])
            
            # Trove storefront data
            trove_storefront = meta.get('trove-storefront', {}).get('data', {})
            structured['brand'] = trove_storefront.get('brand', '')
            structured['color_display'] = trove_storefront.get('colorDisplay', '')
            structured['material'] = trove_storefront.get('material', '')
            structured['details'] = trove_storefront.get('details', [])
            structured['fit'] = trove_storefront.get('fit', '')
            structured['weight'] = trove_storefront.get('weightOz', '')
            
        return structured
    
    def create_searchable_text(self, structured_product: Dict[str, Any]) -> str:
        """
        Create searchable text from structured product data
        
        Args:
            structured_product: Structured product data
            
        Returns:
            Combined text for embedding
        """
        text_parts = []
        
        # Add title and brand
        if structured_product.get('title'):
            text_parts.append(f"Title: {structured_product['title']}")
        if structured_product.get('brand'):
            text_parts.append(f"Brand: {structured_product['brand']}")
        if structured_product.get('vendor'):
            text_parts.append(f"Vendor: {structured_product['vendor']}")
            
        # Add categories and tags
        if structured_product.get('category'):
            text_parts.append(f"Category: {', '.join(structured_product['category'])}")
        if structured_product.get('tags'):
            text_parts.append(f"Tags: {', '.join(structured_product['tags'])}")
            
        # Add product details
        if structured_product.get('color_display'):
            text_parts.append(f"Color: {structured_product['color_display']}")
        if structured_product.get('material'):
            text_parts.append(f"Material: {structured_product['material']}")
        if structured_product.get('fit'):
            text_parts.append(f"Fit: {structured_product['fit']}")
            
        # Add detailed features
        if structured_product.get('details'):
            details_text = ' '.join(structured_product['details'])
            text_parts.append(f"Details: {details_text}")
            
        return ' '.join(text_parts)
    
    def generate_vector_id(self, product_data: Dict[str, Any]) -> str:
        """
        Generate a unique vector ID for the product
        
        Args:
            product_data: Product data
            
        Returns:
            Unique vector ID
        """
        # Use product ID and handle to create unique ID
        unique_string = f"{product_data.get('id', '')}_{product_data.get('handle', '')}"
        return hashlib.md5(unique_string.encode()).hexdigest()
    
    def process_and_store_products(self, json_file_path: str, batch_size: int = 100):
        """
        Process products from JSON file and store in Pinecone
        
        Args:
            json_file_path: Path to the product JSON file
            batch_size: Number of vectors to upsert in each batch
        """
        # Load data
        data = self.load_product_data(json_file_path)
        
        # Get products from results.hits
        products = data.get('results', [{}])[0].get('hits', [])
        print(f"Found {len(products)} products to process")
        
        if not products:
            print("No products found in the data")
            return
        
        # Setup index
        self.setup_index()
        
        # Process products in batches
        vectors_to_upsert = []
        processed_count = 0
        
        for i, product in enumerate(products):
            try:
                # Structure the product data
                structured_product = self.structure_product_data(product)
                
                # Create searchable text
                searchable_text = self.create_searchable_text(structured_product)
                
                # Generate embedding
                embedding = self.model.encode(searchable_text).tolist()
                
                # Create vector ID
                vector_id = self.generate_vector_id(structured_product)
                
                # Prepare metadata (Pinecone has metadata size limits)
                metadata = {
                    'title': structured_product.get('title', '')[:500],  # Limit length
                    'brand': structured_product.get('brand', '')[:100],
                    'vendor': structured_product.get('vendor', '')[:100],
                    'price': structured_product.get('price', 0),
                    'category': structured_product.get('category', [])[:5],  # Limit array size
                    'color_display': structured_product.get('color_display', '')[:100],
                    'department': structured_product.get('department', [])[:3],
                    'product_type': structured_product.get('product_type', '')[:100],
                    'handle': structured_product.get('handle', '')[:200],
                    'processed_at': datetime.now().isoformat()
                }
                
                # Add to batch
                vectors_to_upsert.append({
                    'id': vector_id,
                    'values': embedding,
                    'metadata': metadata
                })
                
                processed_count += 1
                
                # Upsert batch when it reaches batch_size
                if len(vectors_to_upsert) >= batch_size:
                    print(f"Upserting batch of {len(vectors_to_upsert)} vectors...")
                    self.index.upsert(vectors=vectors_to_upsert)
                    vectors_to_upsert = []
                    
                if processed_count % 50 == 0:
                    print(f"Processed {processed_count}/{len(products)} products")
                    
            except Exception as e:
                print(f"Error processing product {i}: {e}")
                continue
        
        # Upsert remaining vectors
        if vectors_to_upsert:
            print(f"Upserting final batch of {len(vectors_to_upsert)} vectors...")
            self.index.upsert(vectors=vectors_to_upsert)
        
        print(f"\nCompleted! Processed {processed_count} products and stored in Pinecone index '{self.index_name}'")
        
        # Print index stats
        stats = self.index.describe_index_stats()
        print(f"Index stats: {stats}")
    
    def search_products(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for products using natural language query
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of matching products with scores
        """
        if not self.index:
            print("Index not initialized. Call setup_index() first.")
            return []
        
        # Generate query embedding
        query_embedding = self.model.encode(query).tolist()
        
        # Search in Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        return results.matches

def main():
    """
    Main function to run the product processing
    """
    # Configuration
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    JSON_FILE_PATH = r"C:\Users\nc157\Projects\dont\task\darzy-ai\algolia_product.json"
    INDEX_NAME = "product-vectors"
    
    if not PINECONE_API_KEY:
        print("Error: PINECONE_API_KEY environment variable not set")
        print("Please set your Pinecone API key in the .env file or environment variables")
        return
    
    if not os.path.exists(JSON_FILE_PATH):
        print(f"Error: JSON file not found at {JSON_FILE_PATH}")
        return
    
    # Initialize processor
    processor = ProductToPinecone(PINECONE_API_KEY, INDEX_NAME)
    
    try:
        # Process and store products
        print("Starting product processing and storage...")
        processor.process_and_store_products(JSON_FILE_PATH)
        
        # Example search
        print("\n" + "="*50)
        print("Testing search functionality:")
        results = processor.search_products("blue jacket patagonia", top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"\nResult {i} (Score: {result.score:.4f}):")
            metadata = result.metadata
            print(f"  Title: {metadata.get('title', 'N/A')}")
            print(f"  Brand: {metadata.get('brand', 'N/A')}")
            print(f"  Price: ${metadata.get('price', 'N/A')}")
            print(f"  Category: {metadata.get('category', 'N/A')}")
            print(f"  Color: {metadata.get('color_display', 'N/A')}")
        
    except Exception as e:
        print(f"Error during processing: {e}")

if __name__ == "__main__":
    main()