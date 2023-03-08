#https://huggingface.co/microsoft/BioGPT-Large
from transformers import pipeline
import torch
import json

from ray import serve

from transformers import BioGptTokenizer, BioGptForCausalLM

#@serve.deployment(route_prefix="/biogpt-large", ray_actor_options={"num_gpus": 1})
@serve.deployment(ray_actor_options={"num_gpus": 1}, health_check_timeout_s=600)
class BioGptLarge:
    def __init__(self):
        self.pipe_biogpt = pipeline("text-generation", model="/mnt/llm-cache/biogpt-large/", device="cuda:0")
        #self.pipe_biogpt = pipeline("text-generation", model="microsoft/BioGPT-Large", device="cuda:0")

        print(f"Is CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    async def __call__(self, starlette_request):
        request = await starlette_request.body()
        data = json.loads(request)
        prompt = data['prompt']
        max_length = data.get('max_length', 100)
        num_sequences = data.get("num_sequences", 5)
        output_biogpt = self.pipe_biogpt(prompt, max_length=max_length, num_return_sequences=num_sequences, do_sample=True)
        result = output_biogpt[0]["generated_text"]
        return result

#biogptmodel = BioGptLarge.bind()
#BIOGPT.deploy()
