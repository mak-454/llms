#https://huggingface.co/microsoft/biogpt
from transformers import pipeline
import torch
import json
from ray import serve


@serve.deployment(ray_actor_options={"num_gpus": 1})
class BioGpt:
    def __init__(self):
        self.pipe_biogpt = pipeline("text-generation", model="/mnt/llm-cache/biogpt/", device="cuda:0")

        print(f"Is CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    async def __call__(self, starlette_request):
        request = await starlette_request.body()
        text = json.loads(request)
        print(text)
        output_biogpt = self.pipe_biogpt(text, max_length=100, num_return_sequences=1)
        result = output_biogpt[0]["generated_text"]
        print(result)
        return {"result": result}
