import requests

query = "What is all_gather_object and how is it different from all_gather?"

params = {
    "query": query
}
r = requests.get("http://localhost:9600/ask", params=params)
resp = r.json()

print(resp['answer'])
print("Sources -", resp['sources'])