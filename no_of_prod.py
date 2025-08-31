import json
data = json.load(open(r"C:\Users\nc157\Projects\dont\task\darzy-ai\algolia_product.json"))
products = data.get('results', [{}])[0].get('hits', [])
print(len(products))
