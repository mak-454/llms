from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM
import time
import json

from ray import serve

#@serve.deployment(route_prefix="/gptjt6bcb", ray_actor_options={"num_gpus": 2}, health_check_timeout_s=600)
#@serve.deployment(route_prefix="/gptj6bcb", health_check_timeout_s=600)
@serve.deployment(ray_actor_options={"num_gpus": 2}, health_check_timeout_s=600)
class GptJ6B:
    def __init__(self):
        #checkpoint = "EleutherAI/gpt-j-6B"
        checkpoint = "/mnt/llm-cache/gpt-j-6B/full-gpt-j-6B"
        config = AutoConfig.from_pretrained(checkpoint)

        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)

        from accelerate import load_checkpoint_and_dispatch

        model = load_checkpoint_and_dispatch(
            model, "/mnt/llm-cache/gpt-j-6B/sharded-gpt-j-6B", device_map="auto", no_split_module_classes=["GPTJBlock"]
        )

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        self.model = model
        self.tokenizer = tokenizer


    async def __call__(self, starlette_request):
        start_time = time.time()

        request = await starlette_request.body()
        #input_text = "Google was founded by"
        data = json.loads(request)

        prompt = data['prompt']
        max_length = data.get('max_length',100)
        temperature = data.get('temperature', 0.8)
         
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to("cuda")
              
        output = self.model.generate(
           input_ids,
           attention_mask=inputs["attention_mask"].to("cuda"),
           do_sample=True,
           max_length=max_length,
           temperature=temperature,
           use_cache=True,
           top_p=0.9
        )
         
        end_time = time.time() - start_time
        print("Total Time => ",end_time)
        gentext = self.tokenizer.decode(output[0])
        return gentext

#gptj6bmodel = gptj6b.bind()
