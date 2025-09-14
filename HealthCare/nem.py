from ollama import Client 

client = Client()
model = client.list()
print(model)