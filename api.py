#!/usr/bin/env python3
"""
API endpoint for querying products from Pinecone vector database
"""

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Union
from dotenv import load_dotenv
from product_to_pinecone import ProductToPinecone

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import (
    create_openai_functions_agent,
    AgentExecutor,
    AgentType,
    initialize_agent,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Product Search API with LangChain Agent",
    description="API for searching products using vector similarity and LangChain-powered conversational AI",
)

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = "product-vectors-example"

# Initialize LangChain components
llm = None
agent_executor = None
if OPENAI_API_KEY:
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo", temperature=0.7)

    # Create product search tool for the agent
    def search_products_tool(query: str) -> List[dict]:
        """Search for products using vector similarity"""
        if not processor:
            return []
        results = processor.search_products(query, top_k=10)
        return [
            {
                "title": r.metadata.get("title", "N/A"),
                "brand": r.metadata.get("brand", "N/A"),
                "price": r.metadata.get("price", "N/A"),
                "color": r.metadata.get("color_display", "N/A"),
                "score": r.score,
                "product_id": r.id,
            }
            for r in results
        ]

    def count_products_tool(query: str) -> str:
        """Return the number of products matching a query"""
        if not processor:
            return "Product search unavailable"
        results = processor.search_products(query, top_k=100)
        return f"Found {len(results)} products for query '{query}'"


from collections import Counter
from typing import List, Optional


def product_attributes_tool(
    attribute: str, query: str, top_k: int = 50, most_common: Optional[int] = 1
) -> List[str]:
    """Return the most common attribute values from matching products"""
    if not processor:
        return []

    results = processor.search_products(query, top_k=top_k)
    attr_list = []

    for r in results:
        val = r.metadata.get(attribute)
        if isinstance(val, list):
            attr_list.extend(val)
        elif val:
            attr_list.append(val)

    if not attr_list:
        return []

    counts = Counter(attr_list)
    return [item for item, _ in counts.most_common(most_common)]


def filter_products_tool(filters: dict, top_k: int = 50) -> List[dict]:
    """
    filters example: {"color_display": "Blue", "brand": "Patagonia", "price_min": 100, "price_max": 200}
    """
    if not processor:
        return []

    query = filters.get("query", "")
    results = processor.search_products(query, top_k=top_k)

    filtered = []
    for r in results:
        metadata = r.metadata
        match = True
        for key, value in filters.items():
            if key == "price_min" and metadata.get("price", 0) < value:
                match = False
            elif key == "price_max" and metadata.get("price", 0) > value:
                match = False
            elif (
                key not in ["price_min", "price_max", "query"]
                and metadata.get(key) != value
            ):
                match = False
        if match:
            filtered.append(
                {
                    "title": metadata.get("title", "N/A"),
                    "brand": metadata.get("brand", "N/A"),
                    "price": metadata.get("price", "N/A"),
                    "color": metadata.get("color_display", "N/A"),
                    "score": r.score,
                    "product_id": r.id,
                }
            )
    return filtered


def recommend_similar_tool(product_id: str, top_k: int = 5) -> List[dict]:
    """Return similar products based on vector similarity"""
    if not processor or not processor.index:
        return []

    # Fetch the vector for the given product_id
    vector_info = processor.index.fetch(ids=[product_id])
    if (
        not vector_info
        or "vectors" not in vector_info
        or product_id not in vector_info["vectors"]
    ):
        return []

    vector = vector_info["vectors"][product_id]["values"]

    # Query Pinecone for similar products
    results = processor.index.query(
        vector=vector, top_k=top_k + 1, include_metadata=True
    )

    # Exclude the original product
    similar = []
    for r in results.matches:
        if r.id == product_id:
            continue
        metadata = r.metadata
        similar.append(
            {
                "title": metadata.get("title", "N/A"),
                "brand": metadata.get("brand", "N/A"),
                "price": metadata.get("price", "N/A"),
                "color": metadata.get("color_display", "N/A"),
                "score": r.score,
                "product_id": r.id,
            }
        )
    return similar


# Define tools for the agent
tools = [
    Tool(
        name="search_products",
        description="Search products by query (title, brand, color, category).",
        func=search_products_tool,
    ),
    Tool(
        name="count_products",
        description="Count the number of products matching a query.",
        func=count_products_tool,
    ),
    Tool(
        name="product_attributes",
        description="Get unique values for a product attribute (color, brand, size, material, category) from matching products.",
        func=product_attributes_tool,
    ),
    Tool(
        name="filter_products",
        description="Filter products using criteria like color, brand, category, or price range.",
        func=filter_products_tool,
    ),
    Tool(
        name="recommend_similar",
        description="Recommend similar products based on a given product ID.",
        func=recommend_similar_tool,
    ),
]


# Create agent prompt
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="""You are a product recommendation assistant.

        - ALWAYS use the search_products tool whenever the user asks about products, prices, brands, colors, quantities, materials, or categories. 
        - DO NOT guess or answer generically when the query is product-related.
        - If the query is NOT product-related, answer normally as a helpful assistant.
        - When presenting results, summarize them conversationally, not just as a list.
        - Use the product_attributes tool to get unique attribute values for filters.
        - Use the filter_products tool to apply multiple filters.
        - Use the recommend_similar tool to get similar products.
        """
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessage(content="{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# Create memory for conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create agent
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True,
)

# Initialize processor
processor = None
if PINECONE_API_KEY:
    processor = ProductToPinecone(PINECONE_API_KEY, INDEX_NAME)
    processor.setup_index()


class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5


class SearchResult(BaseModel):
    title: str
    brand: str
    price: Union[float, str]
    color: str
    score: float
    product_id: str


class AgentRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None


class AgentResponse(BaseModel):
    query: str
    response: str
    conversation_id: Optional[str] = None


@app.get("/")
async def root():
    return {"message": "Product Search API is running"}


@app.post("/search", response_model=List[SearchResult])
async def search_products(request: SearchRequest):
    """
    Search for products using natural language query
    """
    if not processor:
        raise HTTPException(status_code=500, detail="Pinecone API key not configured")

    try:
        results = processor.search_products(request.query, top_k=request.top_k)

        formatted_results = []
        for result in results:
            metadata = result.metadata
            formatted_results.append(
                SearchResult(
                    title=metadata.get("title", "N/A"),
                    brand=metadata.get("brand", "N/A"),
                    price=metadata.get("price", "N/A"),
                    color=metadata.get("color_display", "N/A"),
                    score=result.score,
                    product_id=metadata.get("product_id", result.id),
                )
            )

        return formatted_results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/agent", response_model=AgentResponse)
async def agent_chat(request: AgentRequest):
    """
    LangChain agent for intelligent product recommendations and general assistance
    """
    if not processor:
        raise HTTPException(status_code=500, detail="Pinecone API key not configured")

    if not agent_executor:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")

    try:

        query = request.query
        # Use the agent to process the query
        response = agent_executor.invoke({"input": query})
        print(response)

        return AgentResponse(
            query=query,
            response=response["output"],
            conversation_id=request.conversation_id,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Agent processing failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
