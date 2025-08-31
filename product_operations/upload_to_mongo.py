from pymongo import MongoClient
import json
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client.get_database("patagonia-products")
products_col = db.get_collection("products")

JSON_FILE_PATH = r"C:\Users\nc157\Projects\dont\task\darzy-ai\algolia_product.json"
with open(JSON_FILE_PATH, "r") as f:
    data = json.load(f)
    products = data["results"][0]["hits"]

# Insert products
for p in products:
    p["_id"] = str(p.get("id"))  # use product id as _id
    products_col.replace_one({"_id": p["_id"]}, p, upsert=True)

print(f"Inserted {len(products)} products")


# class UploadToMongo:
#     def __init__(self):
#         self.client = MongoClient(os.getenv("MONGO_URI"))
#         self.db = self.client.get_database("patagonia-products")
#         self.products_col = self.db.get_collection("products")

#     def upload_products(self, json_file_path: str):
#         with open(json_file_path, "r") as f:
#             data = json.load(f)
#             products = data["results"][0]["hits"]

#         # Insert products
#         for p in products:
#             p["_id"] = str(p.get("id"))  # use product id as _id
#             self.products_col.replace_one({"_id": p["_id"]}, p, upsert=True)

#         print(f"Inserted {len(products)} products")

#     def product_col(self):
#         return self.products_col

#     def close(self):
#         self.client.close()