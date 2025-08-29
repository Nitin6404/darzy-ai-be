# Product to Pinecone Vector Database with LangChain Agent

This module processes Algolia product JSON data, structures it properly, and stores it in a Pinecone vector database for semantic search and retrieval. It includes a LangChain-powered conversational agent for intelligent product recommendations.

## Features

- **Data Structuring**: Extracts and organizes key product information from Algolia JSON format
- **Vector Embeddings**: Generates semantic embeddings using SentenceTransformers
- **Pinecone Integration**: Stores vectors in Pinecone for fast similarity search
- **Batch Processing**: Efficiently processes large product datasets
- **Semantic Search**: Enables natural language product search
- **LangChain Agent**: Conversational AI agent with product search tools
- **Memory & Context**: Maintains conversation history for better recommendations

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

Create a `.env` file in the project root or set environment variables:

```bash
PINECONE_API_KEY=your_pinecone_api_key_here
```

### 3. Run the Processor

```python
from product_to_pinecone import ProductToPinecone

# Initialize
processor = ProductToPinecone(api_key="your_api_key", index_name="products")

# Process and store products
processor.process_and_store_products("path/to/algolia_product.json")

# Search products
results = processor.search_products("blue patagonia jacket", top_k=5)
```

### 4. Run Examples

```bash
# Basic product processing
python example_usage.py

# Start API server
python run_api.py

# Test LangChain agent
python example_agent.py
```

## File Structure

```
rag/
├── product_to_pinecone.py    # Main processor class
├── api.py                    # FastAPI server with LangChain agent
├── example_usage.py          # Basic usage examples
├── example_agent.py          # LangChain agent examples
├── run_api.py               # API server runner
├── requirements.txt          # Dependencies
└── README.md                # This file
```

## How It Works

### 1. Data Structuring

The processor extracts key information from Algolia product data:

- **Basic Info**: Title, vendor, price, tags
- **Product Details**: Category, color, material, fit
- **Metadata**: Brand, department, condition, size
- **Features**: Product details and specifications

### 2. Text Generation

Creates searchable text by combining:
- Product title and brand
- Categories and tags
- Color and material information
- Detailed product features

### 3. Vector Embedding

Uses `all-MiniLM-L6-v2` model to generate 384-dimensional embeddings that capture semantic meaning.

### 4. Pinecone Storage

Stores vectors with metadata in Pinecone index for fast retrieval:

```python
{
    'id': 'unique_product_id',
    'values': [0.1, 0.2, ...],  # 384-dim embedding
    'metadata': {
        'title': 'Product Title',
        'brand': 'Brand Name',
        'price': 199.99,
        'category': ['Jackets'],
        'color_display': 'Navy Blue'
    }
}
```

## Configuration Options

### ProductToPinecone Parameters

- `pinecone_api_key`: Your Pinecone API key
- `index_name`: Name of the Pinecone index (default: "product-vectors")

### Processing Parameters

- `batch_size`: Number of vectors to upsert per batch (default: 100)
- `dimension`: Vector dimension (default: 384)
- `metric`: Distance metric (default: "cosine")

## Search Examples

### Vector Search

```python
# Initialize processor
processor = ProductToPinecone(api_key, "my-products")

# Natural language searches
results = processor.search_products("waterproof hiking jacket", top_k=5)
results = processor.search_products("warm winter coat for men", top_k=3)
results = processor.search_products("blue patagonia down sweater", top_k=10)

# Process results
for result in results:
    print(f"Product: {result.metadata['title']}")
    print(f"Score: {result.score}")
    print(f"Price: ${result.metadata['price']}")
```

### LangChain Agent

```python
# Using the API endpoint
import requests

response = requests.post("http://localhost:8000/agent", json={
    "query": "I need a warm jacket for winter hiking",
    "conversation_id": "optional-session-id"
})

result = response.json()
print(f"Agent: {result['response']}")
```

### cURL Examples

```bash
# Vector search
curl -X POST "http://localhost:8000/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "blue patagonia jacket", "top_k": 5}'

# LangChain agent
curl -X POST "http://localhost:8000/agent" \
     -H "Content-Type: application/json" \
     -d '{"query": "I need a warm jacket for hiking"}'
```

## Data Format

### Input (Algolia JSON)

Expects JSON structure:

```json
{
  "results": [
    {
      "hits": [
        {
          "id": "product_id",
          "title": "Product Title",
          "vendor": "Brand Name",
          "price": 199.99,
          "meta": {
            "trove-facets": {...},
            "trove-storefront": {...}
          }
        }
      ]
    }
  ]
}
```

### Output (Pinecone Vectors)

Stores structured vectors with rich metadata for semantic search.

## Error Handling

- **Missing API Key**: Clear error message with setup instructions
- **File Not Found**: Validates JSON file existence
- **Processing Errors**: Continues processing other products if one fails
- **Index Issues**: Automatic index creation if it doesn't exist

## Performance

- **Batch Processing**: Processes products in configurable batches
- **Progress Tracking**: Shows processing progress every 50 products
- **Memory Efficient**: Processes data in chunks to avoid memory issues
- **Fast Search**: Leverages Pinecone's optimized vector search

## Troubleshooting

### Common Issues

1. **"PINECONE_API_KEY not found"**
   - Set the environment variable or add to .env file

2. **"JSON file not found"**
   - Check the file path in the script
   - Ensure the algolia_product.json file exists

3. **"Index creation failed"**
   - Verify Pinecone API key is valid
   - Check Pinecone account limits

4. **"Embedding model download"**
   - First run downloads the SentenceTransformer model
   - Ensure internet connection for model download

### Performance Tips

- Increase `batch_size` for faster processing (up to 1000)
- Use smaller embedding models for faster processing
- Monitor Pinecone usage limits and quotas

## Dependencies

### Core Dependencies
- `pinecone-client`: Pinecone vector database client
- `sentence-transformers`: Text embedding generation
- `torch`: PyTorch for model inference
- `transformers`: Hugging Face transformers library
- `python-dotenv`: Environment variable management

### API Dependencies
- `fastapi`: Web API framework
- `uvicorn`: ASGI server
- `pydantic`: Data validation

### LangChain Dependencies
- `langchain`: Core LangChain framework
- `langchain-openai`: OpenAI integration for LangChain
- `langchain-community`: Community tools and integrations

## License

This module is part of the Brand DNA extraction system.