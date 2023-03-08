import requests

url = "http://127.0.0.1:8000/mpnetbasev2"
sentences = ["This is an example sentence", "Each sentence is converted"]
resp = requests.post(url, json=sentences)
print(resp.content)
