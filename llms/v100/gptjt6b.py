from ray import serve
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import json
import re
import ast

#@serve.deployment(name="gptjt6bv100", route_prefix="/gptjt6b", ray_actor_options={"num_gpus": 1, "accelerator_type": "V100"}, health_check_timeout_s=600)
@serve.deployment(name="gptjt6bv100", ray_actor_options={"num_gpus": 1, "accelerator_type": "V100"}, health_check_timeout_s=600)
class GptJT6Bv100():
    def __init__(self):
        #tokenizer = AutoTokenizer.from_pretrained("togethercomputer/GPT-JT-6B-v1")
        self.tokenizer = AutoTokenizer.from_pretrained("/mnt/llm-cache/gptjt6b")
        self.model = AutoModelForCausalLM.from_pretrained("/mnt/llm-cache/gptjt6b", torch_dtype=torch.float16, low_cpu_mem_usage=True).to("cuda")

    async def __call__(self, starlette_request):
        request = await starlette_request.body()
        data = json.loads(request)
        stime = time.time()

        prompt = data.pop('prompt', "")
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to("cuda")

        kwargs = data.pop("model_kwargs", {})
        try:
            output = self.model.generate(input_ids,
                    attention_mask = inputs["attention_mask"].to("cuda"),
                    **kwargs)
        except ValueError as ve:
            ves = str(ve)
            print("The original string is : " + ves)
            # Extract substrings between brackets
            # Using regex
            res = re.findall(r'\[.*?\]', ves)
            arr = ast.literal_eval(res[0])
            print(arr)
            for elem in arr:
                #pop out unused parameters
                val = kwargs.pop(elem, None)
                print(f"Popped out unused parameter {elem} for this model")

            output = self.model.generate(input_ids,
                    attention_mask = inputs["attention_mask"].to("cuda"),
                    **data)


        gentext = self.tokenizer.decode(output[0].tolist())
        print("--- %s seconds ---" % (time.time() - stime))
        return gentext

#GPTJT6BV100.deploy()
#gptjt6bv100 = GPTJT6BV100.bind()


'''
EXAMPLES = [
"""Extract all the names of people, places, and organizations from the following sentences.

Sentence: Satya Nadella, the CEO of Microsoft, was visiting the Bahamas last May.
Entities: Satya Nadella, Microsoft, Bahamas

Sentence: Pacific Northwest cities include Seattle and Portland, which I have visited with Vikash.
Entities: Vikash, Seattle,""",

"""In a shocking finding, scientists discovered a herd of unicorns living in a remote,
         previously unexplored valley, in the Andes Mountains. Even more surprising to the
          researchers was the fact that the unicorns spoke perfect English.""",

"""Label whether the following tweet contains hate speech against either immigrants or women. Hate Speech (HS) is commonly defined as any communication that disparages a person or a group on the basis of some characteristic such as race, color, ethnicity, gender, sexual orientation, nationality, religion, or other characteristics.
Possible labels:
1. hate speech
2. not hate speech

Tweet: HOW REFRESHING! In South Korea, there is no such thing as 'political correctness" when it comes to dealing with Muslim refugee wannabes via @user
Label: hate speech

Tweet: New to Twitter-- any men on here know what the process is to get #verified?
Label: not hate speech

Tweet: Dont worry @user you are and will always be the most hysterical woman.
Label:"""
]
'''
