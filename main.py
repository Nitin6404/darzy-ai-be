import os
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Dict
from dotenv import load_dotenv
from product_operations.product_search import ProductSearch

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not set in .env")

app = FastAPI(title="Product Search API")

# Initialize search client
search_client = ProductSearch(PINECONE_API_KEY)

class AnswerResponse(BaseModel):
    query: str
    answer: str
    retrieved_products: List[Dict]

def build_filters(
    brand: Optional[str] = None,
    color: Optional[str] = None,
    category: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None
) -> Optional[dict]:
    filters = {}
    if brand: filters["brand"] = {"$eq": brand}
    if color: filters["color"] = {"$in": [color]}
    if category: filters["category"] = {"$in": [category]}
    price_filter = {}
    if min_price is not None: price_filter["$gte"] = min_price
    if max_price is not None: price_filter["$lte"] = max_price
    if price_filter: filters["price"] = price_filter
    return filters if filters else None

# Define standard response model
# def standard_response(data: List[Dict], query: str, top_k: int, filters: Optional[dict] = None):
#     return {
#         "query": query,
#         "top_k": top_k,
#         "filters": filters or {},
#         "results_count": len(data),
#         "results": data
#     }

@app.get("/")
def root():
    return {"message": "Welcome to the Product Search API!"}

# @app.get("/search")
# def search_products(
#     query: str = Query(..., description="Search query string"),
#     top_k: int = Query(5, description="Number of results to return"),
#     brand: Optional[str] = Query(None, description="Filter by brand"),
#     color: Optional[str] = Query(None, description="Filter by color")
# ):
#     # Build filter dict if provided
#     filters = {}
#     if brand:
#         filters["brand"] = {"$eq": brand}
#     if color:
#         filters["color"] = {"$in": [color]}

#     try:
#         results = search_client.query_products(query=query, top_k=top_k, filters=filters or None)
#         return standard_response(results, query, top_k, filters)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@app.get("/answer")
async def get_answer(
    query: str = Query(..., description="User query for product recommendation"),
    top_k: int = Query(5, description="Number of products to use as context"),
    brand: Optional[str] = Query(None),
    color: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    min_price: Optional[float] = Query(None),
    max_price: Optional[float] = Query(None)
):
    filters = build_filters(brand=brand, color=color, category=category, min_price=min_price, max_price=max_price)

    try:
        # Use async version if implemented
        products = search_client.query_products(query, top_k=top_k, filters=filters or None)
        if not products:
            return {"answer": "No matching products found.", "retrieved_products": []}

        answer = search_client.generate_answer(query, top_k=top_k, filters=filters or None)

        return AnswerResponse(query=query, answer=answer, retrieved_products=products)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
