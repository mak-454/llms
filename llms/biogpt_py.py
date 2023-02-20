#https://huggingface.co/microsoft/BioGPT-Large-PubMedQA
#https://huggingface.co/spaces/katielink/biogpt-qa-demo
from transformers import pipeline
import torch
import json

from ray import serve
import ray

# Connect to the running Ray cluster.
#ray.init(address="auto", namespace="default")
# Bind on 0.0.0.0 to expose the HTTP server on external IPs.
#serve.start(detached=True, http_options={"host": "0.0.0.0"})


pipe_biogpt = pipeline("text-generation", model="/mnt/llm-cache/biogpt-large/", device="cuda:0")
#pipe_biogpt = pipeline("text-generation", model="microsoft/BioGPT-Large", device="cuda:0")

print(f"Is CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")

text = "hi"
print(text)
output_biogpt = pipe_biogpt(text, max_length=100, num_return_sequences=1)
result = output_biogpt[0]["generated_text"]
print(result)
