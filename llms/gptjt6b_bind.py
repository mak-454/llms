from transformers import pipeline
import torch

from ray import serve
from transformers import AutoTokenizer

from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import load_checkpoint_and_dispatch

@serve.deployment(route_prefix="/gptjt6b", ray_actor_options={"num_gpus": 2}, health_check_timeout_s=600)
class GPTJT6B:
    def __init__(self):
        print(f"Is CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        self.model = None

        #checkpoint = "togethercomputer/GPT-JT-6B-v1"
        checkpoint = "/home/ray/gpt-jt-6b-v1-sharded"
        config = AutoConfig.from_pretrained(checkpoint)

        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_config(config)

        print("111111111111111")

        self.model = load_checkpoint_and_dispatch(
            self.model, "/home/ray/gpt-jt-6b-v1-sharded/", device_map="auto", no_split_module_classes=["GPTJBlock"])

        print(self.model.hf_device_map)
        self.tokenizer = AutoTokenizer.from_pretrained("/home/ray/gpt-jt-6b-v1-sharded/")

    async def __call__(self, starlette_request):
        prompt = "In a shocking finding, scientists discovered a herd of unicorns living in a remote, " \
                 "previously unexplored valley, in the Andes Mountains. Even more surprising to the " \
                  "researchers was the fact that the unicorns spoke perfect English."
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to(0)
        output = self.model.generate(inputs["input_ids"])
        gentext = self.tokenizer.decode(output[0].tolist())
        print(gentext)
        return {"result": gentext}

gptjt6bmodel = GPTJT6B.bind()
