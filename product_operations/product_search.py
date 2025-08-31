import os
import json
from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

class ProductSearch:
    def __init__(self, pinecone_api_key: str, index_name: str = "product-vectors"):
        """
        Initialize Pinecone search client
        """
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index_name = index_name
        self.embedder = OpenAIEmbeddings(model="text-embedding-3-large")
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        self.index = None
        self.setup_index()

    def setup_index(self, dimension: int = 3072, metric: str = "cosine"):
        """Connect to existing Pinecone index"""
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        if self.index_name not in existing_indexes:
            raise ValueError(f"Index '{self.index_name}' does not exist. Please create it first.")
        self.index = self.pc.Index(self.index_name)

    def get_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for search query"""
        return self.embedder.embed_query(query)

    def search(
        self, 
        query_embedding: List[float], 
        top_k: int = 5, 
        filters: dict = None
    ) -> List[Dict[str, Any]]:
        """Search Pinecone with optional metadata filters"""
        query_args = {
            "vector": query_embedding,
            "top_k": top_k,
            "include_metadata": True
        }
        if filters:
            query_args["filter"] = filters
        results = self.index.query(**query_args)
        return results.matches

    def format_results(self, matches: List[Dict[str, Any]]) -> List[dict]:
        """Clean search results"""
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
    ) -> List[dict]:
        """One-line search method with optional filters"""
        embedding = self.get_query_embedding(query)
        matches = self.search(embedding, top_k=top_k, filters=filters)
        return self.format_results(matches)

    def generate_answer(self, query: str, top_k: int = 5, filters: dict = None) -> str:
        # Step 1: Retrieve top-k products
        products = self.query_products(query, top_k=top_k, filters=filters)
        if not products:
            return "No matching products found. Try different keywords or filters."

        # Step 2: Convert products to structured JSON
        product_context_json = json.dumps(products, indent=2)

        # Step 3: Create prompt template
        system_prompt = (
            "You are a product recommendation assistant. "
            "Answer user queries concisely using only the provided product data. "
            "Highlight best matches, compare them briefly, and do not hallucinate."
        )

        human_prompt = "User query: {query}\nProducts:\n{context}\nAnswer the query using only these products."

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])

        # Step 4: Chain LLM with parser
        chain = prompt_template | self.llm | StrOutputParser()

        # Step 5: Invoke chain
        response = chain.invoke({"context": product_context_json, "query": query})
        return response
