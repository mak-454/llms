#https://huggingface.co/microsoft/BioGPT-Large-PubMedQA
#https://huggingface.co/spaces/katielink/biogpt-qa-demo
from transformers import pipeline
import torch
import json

from ray import serve

#@serve.deployment(route_prefix="/biogpt-large-pubmedqa", ray_actor_options={"num_gpus": 1})
@serve.deployment(ray_actor_options={"num_gpus": 1}, health_check_timeout_s=600)
class BioGPTLargePubmedQA:
    def __init__(self):
        #pipe_biogpt = pipeline("text-generation", model="microsoft/BioGPT-Large-PubMedQA", device="cuda:0")
        self.pipe_biogpt = pipeline("text-generation", model="/mnt/llm-cache/BioGPT-Large-PubMedQA", device="cuda:0")

        print(f"Is CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    async def __call__(self, starlette_request):
        request = await starlette_request.body()
        data = json.loads(request)
        prompt = data['prompt']
        max_length = data.get('max_length',100)
        output_biogpt = self.pipe_biogpt(prompt, max_length=max_length)
        output_biogpt = output_biogpt[0][0]["generated_text"]
        result = output_biogpt.split(' ')[-1]
        return result

''' Examples which work
        examples = [['question: Should chest wall irradiation be included after mastectomy and negative node breast cancer? context: This study aims to evaluate local failure patterns in node negative breast cancer patients treated with post-mastectomy radiotherapy including internal mammary chain only. Retrospective analysis of 92 internal or central-breast node-negative tumours with mastectomy and external irradiation of the internal mammary chain at the dose of 50 Gy, from 1994 to 1998. Local recurrence rate was 5 % (five cases). Recurrence sites were the operative scare and chest wall. Factors associated with increased risk of local failure were age<or = 40 years and tumour size greater than 20mm, without statistical significance. answer: Post-mastectomy radiotherapy should be discussed for a sub-group of node-negative patients with predictors factors of local failure such as age<or = 40 years and larger tumour size. target: the answer to the question given the context is'],
                    ['question: Do some U.S. states have higher/lower injury mortality rates than others? context: This article examines the hypothesis that the six U.S. states with the highest rates of road traffic deaths (group 1 states) also had above-average rates of other forms of injury such as falling, poisoning, drowning, fire, suffocation, homicide, and suicide, and also for the retail trade and construction industries. The converse, second hypothesis, for the six states with the lowest rates of road traffic deaths (group 2 states) is also examined. Data for these 12 states for the period 1983 to 1995 included nine categories of unintentional and four categories of intentional injury. Seventy-four percent of the group 1 states conformed to the first hypothesis, and 85% of the group 2 states conformed to the second hypothesis. answer: Group 1 states are likely to exhibit above-average rates for most other categories of injury death, whereas group 2 states are even more likely to exhibit below-average rates for most other categories of injury death. target: the answer to the question given the context is']]
'''

