# app.py
import streamlit as st
from product_operations.product_search import ProductSearch
import os
from dotenv import load_dotenv

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    st.error("PINECONE_API_KEY not set in .env")
    st.stop()

search_client = ProductSearch(PINECONE_API_KEY)

st.set_page_config(page_title="Product Search & Recommendations", layout="wide")
st.title("🔎 Product Search & Recommendation")

# --- Sidebar Filters ---
st.sidebar.header("Filters")
brand_filter = st.sidebar.text_input("Brand")
color_filter = st.sidebar.text_input("Color")
category_filter = st.sidebar.text_input("Category")
min_price = st.sidebar.number_input("Min Price", min_value=0.0, value=0.0)
max_price = st.sidebar.number_input("Max Price", min_value=0.0, value=0.0)
top_k = st.sidebar.slider("Number of products to retrieve", 1, 10, 5)

query = st.text_input("Enter your product query", "")

if st.button("Search & Recommend") and query:
    with st.spinner("Searching products and generating answer..."):
        filters = {}
        if brand_filter: filters["brand"] = {"$eq": brand_filter}
        if color_filter: filters["color"] = {"$in": [color_filter]}
        if category_filter: filters["category"] = {"$in": [category_filter]}
        if min_price > 0 or max_price > 0:
            price_filter = {}
            if min_price > 0: price_filter["$gte"] = min_price
            if max_price > 0: price_filter["$lte"] = max_price
            filters["price"] = price_filter

        try:
            products = search_client.query_products(query=query, top_k=top_k, filters=filters or None)
            answer = search_client.generate_answer(query=query, top_k=top_k, filters=filters or None)

            st.subheader("💡 Recommendation Answer")
            st.write(answer)

            if products:
                st.subheader("📦 Retrieved Products")
                for p in products:
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        if p.get("product_image"):
                            st.image(p["product_image"], use_column_width=True)
                        else:
                            st.write("No image")
                    with col2:
                        st.markdown(f"**{p.get('title')}**")
                        st.markdown(f"- Brand: {p.get('brand')}")
                        st.markdown(f"- Price: {p.get('price')}")
                        st.markdown(f"- Category: {', '.join(p.get('category', []))}")
                        st.markdown(f"- Color: {', '.join(p.get('color', []))}")
                        st.markdown("---")
            else:
                st.info("No products found matching your query and filters.")

        except Exception as e:
            st.error(f"Error: {e}")
