import json
from typing import List, Dict, Any
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .upload_to_mongo import UploadToMongo
# from upload_to_mongo import UploadToMongo

class ProductSearch:
    def __init__(self, pinecone_api_key: str, index_name: str = "product-vectors"):
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index_name = index_name
        self.embedder = OpenAIEmbeddings(model="text-embedding-3-large")
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        self.index = None
        self.setup_index()

    def setup_index(self):
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        if self.index_name not in existing_indexes:
            raise ValueError(f"Index '{self.index_name}' does not exist.")
        self.index = self.pc.Index(self.index_name)

    def get_query_embedding(self, query: str) -> List[float]:
        return self.embedder.embed_query(query)

    def search(self, query_embedding: List[float], top_k: int = 5, filters: dict = None) -> List[Dict[str, Any]]:
        query_args = {"vector": query_embedding, "top_k": top_k, "include_metadata": True}
        if filters:
            query_args["filter"] = filters
        results = self.index.query(**query_args)
        return results.matches

    def format_results(self, matches: List[Dict[str, Any]]) -> List[dict]:
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

    def get_products_by_ids(self, product_ids: List[str]) -> List[Dict[str, Any]]:
        return list(products_col.find({"_id": {"$in": product_ids}}))

    def query_products(self, query: str, top_k: int = 5, filters: dict = None) -> List[dict]:
        embedding = self.get_query_embedding(query)
        matches = self.search(embedding, top_k=top_k, filters=filters)
        product_ids = [m.id for m in matches]
        return self.get_products_by_ids(product_ids)

    def generate_answer(self, query: str, top_k: int = 5, filters: dict = None) -> str:
        products = self.query_products(query, top_k=top_k, filters=filters)
        if not products:
            return "No matching products found. Try different keywords or filters."

        # leys save product into json file
        with open("products.json", "w") as f:
            json.dump(products, f, indent=2)

        product_context_json = json.dumps(products, indent=2)
        system_prompt = (
            "You are a product recommendation assistant. "
            "Answer user queries concisely using only the provided product data. "
            "Highlight best matches, compare briefly, do not hallucinate."
        )
        human_prompt = "User query: {query}\nProducts:\n{context}\nAnswer using only these products."

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
        chain = prompt_template | self.llm | StrOutputParser()
        response = chain.invoke({"context": product_context_json, "query": query})
        return response
