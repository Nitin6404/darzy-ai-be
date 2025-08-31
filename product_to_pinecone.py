import json
import os
from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import hashlib
from datetime import datetime
from langchain_openai import OpenAIEmbeddings

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
        self.embedder = OpenAIEmbeddings(model="text-embedding-3-large")
        self.index = None
        
    def setup_index(self, dimension: int = 3072, metric: str = "cosine"):
        """
        Create or connect to Pinecone index
        
        Args:
            dimension: Vector dimension (3072 for text-embedding-3-large)
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
    
    def structure_product_data(self, product: dict) -> dict:
        """
        Process raw product data into:
        1. structured metadata (for filters & faceting)
        2. flattened text (for embeddings)
        """
        structured = {} # like object from js

        # Core identity
        structured['id'] = str(product.get('id', ''))
        structured['title'] = product.get('title', '')
        structured['vendor'] = product.get('vendor', '')
        structured['brand'] = product.get('meta', {}).get('trove-storefront', {}).get('data', {}).get('brand', '')
        structured['product_type'] = product.get('product_type', '')
        structured['collections'] = product.get('collections', [])
        structured['sku'] = product.get('sku', '')
        structured['variant_title'] = product.get('variant_title', '')

        # Pricing
        structured['price'] = product.get('price', 0)
        structured['compare_at_price'] = product.get('compare_at_price', 0)
        structured['price_range'] = product.get('price_range', '')
        structured['price_ratio'] = product.get('price_ratio', 0)
        structured['variants_min_price'] = product.get('variants_min_price', 0)
        structured['variants_max_price'] = product.get('variants_max_price', 0)

        # Facets (color, size, gender, condition, etc.)
        facets = product.get('meta', {}).get('trove-facets', {})
        structured['category'] = facets.get('category', [])
        structured['color'] = facets.get('color', [])
        structured['department'] = facets.get('department', [])
        structured['gender'] = facets.get('gender', [])
        structured['size'] = facets.get('size', [])
        structured['condition'] = facets.get('condition', [])

        # Storefront data (detailed info)
        storefront = product.get('meta', {}).get('trove-storefront', {}).get('data', {})
        structured['color_display'] = storefront.get('colorDisplay', '')
        structured['material'] = storefront.get('material', '')
        structured['fit'] = storefront.get('fit', '')
        structured['details'] = storefront.get('details', [])
        structured['weight'] = storefront.get('weightOz', '')

        # Tags
        structured['tags'] = product.get('tags', [])
        structured['named_tags'] = product.get('named_tags', {})
        structured['named_tags_names'] = product.get('named_tags_names', [])

        # Options
        structured['options'] = product.get('options', {})
        structured['option_names'] = product.get('option_names', [])

        # Images
        structured['product_image'] = product.get('product_image', product.get('image', ''))

        # Timestamps
        structured['created_at'] = product.get('created_at', '')
        structured['updated_at'] = product.get('updated_at', '')
        structured['published_at'] = product.get('published_at', '')

        return structured # returing the object

    def product_to_embedding_text(self, structured: dict) -> str: # returning a string
        """Convert structured product metadata into a natural descriptive string for embeddings"""
        parts = []

        # Identity
        if structured.get("title"):
            parts.append(structured["title"])
        if structured.get("brand"):
            parts.append(f"by {structured['brand']}")
        if structured.get("vendor"):
            parts.append(f"Sold by {structured['vendor']}")
        if structured.get("product_type"):
            parts.append(f"Type: {structured['product_type']}")
        if structured.get("variant_title"):
            parts.append(f"Variant: {structured['variant_title']}")

        # Category & segmentation
        if structured.get("category"):
            parts.append("Category: " + ", ".join(structured["category"]))
        if structured.get("department"):
            parts.append("Department: " + ", ".join(structured["department"]))
        if structured.get("gender"):
            parts.append("Gender: " + ", ".join(structured["gender"]))

        # Attributes
        if structured.get("color_display"):
            parts.append(f"Color: {structured['color_display']}")
        elif structured.get("color"):
            parts.append("Color: " + ", ".join(structured["color"]))
        if structured.get("size"):
            parts.append("Size: " + ", ".join(structured["size"]))
        if structured.get("condition"):
            parts.append("Condition: " + ", ".join(structured["condition"]))
        if structured.get("fit"):
            parts.append(f"Fit: {structured['fit']}")

        # Material & features
        if structured.get("material"):
            parts.append(f"Material: {structured['material']}")
        if structured.get("details"):
            parts.append("Features: " + "; ".join(structured["details"]))

        # Options
        if structured.get("option_names"):
            parts.append("Options: " + ", ".join(structured["option_names"]))

        # Tags
        if structured.get("named_tags_names"):
            parts.append("Tags: " + ", ".join(structured["named_tags_names"]))

        # Collections
        if structured.get("collections"):
            parts.append("Collections: " + ", ".join(structured["collections"]))

        # Price
        if structured.get("price_range"):
            parts.append(f"Price Range: {structured['price_range']}")

        return ". ".join(parts) + "."
    
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
    
    def safe_metadata(self, structured: dict) -> dict:
        """Trim structured product into Pinecone-safe metadata"""
        return {
            "title": structured.get("title", "")[:300],
            "brand": structured.get("brand", "")[:100],
            "vendor": structured.get("vendor", "")[:100],
            "price": structured.get("price", 0),
            "category": structured.get("category", [])[:5],
            "color": structured.get("color", [])[:3],
            "size": structured.get("size", [])[:3],
            "condition": structured.get("condition", [])[:2],
            "product_type": structured.get("product_type", "")[:100],
            "processed_at": datetime.now().isoformat()
        }

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
                structured_product = self.structure_product_data(product) # getting a object

                embedding_text = self.product_to_embedding_text(structured_product) # getting a string
                
                # Generate embedding
                embedding = self.embedder.embed_query(embedding_text) # getting a vector
                
                # Create vector ID
                vector_id = self.generate_vector_id(structured_product) # getting a vector id
                
                # Prepare metadata (Pinecone has metadata size limits)
                metadata = self.safe_metadata(structured_product)
                
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
    
    def get_query_embedding(self, query: str) -> List[float]:
        """Return vector for a natural language query"""
        return self.embedder.embed_query(query)

    def search(
        self, 
        query_embedding: List[float], 
        top_k: int = 5, 
        filters: dict = None
    ) -> List[Dict[str, Any]]:
        """
        Search Pinecone with optional metadata filters.
        """
        query_args = {
            "vector": query_embedding,
            "top_k": top_k,
            "include_metadata": True
        }
        if filters:
            query_args["filter"] = filters

        results = self.index.query(**query_args)
        return results.matches

    def format_search_results(self, matches: List[Dict[str, Any]]) -> List[dict]:
        """Return search results in clean, consistent format"""
        formatted = []
        for m in matches:
            formatted.append({
                "id": m.id,
                "score": m.score,
                "title": m.metadata.get("title"),
                "brand": m.metadata.get("brand"),
                "price": m.metadata.get("price"),
                "category": m.metadata.get("category"),
                "color": m.metadata.get("color")
            })
        return formatted

    def query_products(
        self,
        query: str,
        top_k: int = 5,
        filters: dict = None
    ) -> list[dict]:
        """
        One-line method to query products using natural language + optional filters.
        Returns formatted results.
        """
        embedding = self.get_query_embedding(query)
        matches = self.search(embedding, top_k=top_k, filters=filters)
        return self.format_search_results(matches)

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
        filters = {"brand": {"$eq": "Patagonia"}, "color": {"$in": ["blue"]}}
        results = processor.query_products("jacket", top_k=5, filters=filters)
        
        # for i, result in enumerate(results, 1):
        #     print(f"\nResult {i} (Score: {result.score:.4f}):")
        #     metadata = result.metadata
        #     print(f"  Title: {metadata.get('title', 'N/A')}")
        #     print(f"  Brand: {metadata.get('brand', 'N/A')}")
        #     print(f"  Price: ${metadata.get('price', 'N/A')}")
        #     print(f"  Category: {metadata.get('category', 'N/A')}")
        #     print(f"  Color: {metadata.get('color_display', 'N/A')}")

        for r in results:
            print(r)
        
    except Exception as e:
        print(f"Error during processing: {e}")

if __name__ == "__main__":
    main()