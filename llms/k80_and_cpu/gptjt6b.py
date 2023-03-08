#https://huggingface.co/togethercomputer/GPT-JT-6B-v1/tree/main
from transformers import pipeline
import torch
import json

from ray import serve
from transformers import AutoTokenizer

#@serve.deployment(route_prefix="/gptjt6b", ray_actor_options={"num_gpus": 2})
@serve.deployment(ray_actor_options={"num_gpus": 2}, health_check_timeout_s=600)
class GptJT6B:
    def __init__(self):
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoModelForCausalLM

        #checkpoint = "togethercomputer/GPT-JT-6B-v1"
        checkpoint = "/mnt/llm-cache/gpt-jt-6b-v1-sharded"
        config = AutoConfig.from_pretrained(checkpoint)

        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_config(config)

        from accelerate import load_checkpoint_and_dispatch

        self.model = load_checkpoint_and_dispatch(
            self.model, "/mnt/llm-cache/gpt-jt-6b-v1-sharded", device_map="auto", no_split_module_classes=["GPTJBlock"])


        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        print(self.model.hf_device_map)

        print(f"Is CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    async def __call__(self, starlette_request):
        '''
        prompt = "In a shocking finding, scientists discovered a herd of unicorns living in a remote, " \
                 "previously unexplored valley, in the Andes Mountains. Even more surprising to the " \
                  "researchers was the fact that the unicorns spoke perfect English."
        '''
        request = await starlette_request.body()
        data = json.loads(request)
        prompt = data['prompt']
        temperature = data.get('temperature', 0.8)
        max_length = data.get('max_length', 100)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to(0)
        output = self.model.generate(inputs["input_ids"],
                                     do_sample=True,
                                     max_length=max_length,
                                     temperature=temperature,
                                     use_cache=True,
                                     top_p=0.9)
        gentext = self.tokenizer.decode(output[0].tolist())
        return gentext

#GPTJT6B.deploy()
#gptjt6b = GPTJT6B.bind()
