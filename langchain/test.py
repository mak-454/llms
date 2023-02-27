from langchain.llms.dkubeai import HuggingFaceHub
hf = HuggingFaceHub(dkubeai_api_token="invincible", repo_id="EleutherAI/gpt-j-6B")
print(hf("Zoe Kwan is a 20-year old singer and songwriter who has taken Hong Kong’s music scene by storm."))





#========== Trials ============

'''
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
import requests

class CustomLLM(LLM):

    n: int

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        data = ["question: Should chest wall irradiation be included after mastectomy and negative node breast cancer? context: This study aims to evaluate local failure patterns in node negative breast cancer patients treated with post-mastectomy radiotherapy including internal mammary chain only. Retrospective analysis of 92 internal or central-breast node-negative tumours with mastectomy and external irradiation of the internal mammary chain at the dose of 50 Gy, from 1994 to 1998. Local recurrence rate was 5 % (five cases). Recurrence sites were the operative scare and chest wall. Factors associated with increased risk of local failure were age<or = 40 years and tumour size greater than 20mm, without statistical significance. answer: Post-mastectomy radiotherapy should be discussed for a sub-group of node-negative patients with predictors factors of local failure such as age<or = 40 years and larger tumour size. target: the answer to the question given the context is"]
        data = {"prompt": prompt, "max_length": 150, "temperature": 0.8}
        url = "http://127.0.0.1:8000/gptj6b"
        resp = requests.post(url, json=data)
        return resp.content

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"n": self.n}

llm = CustomLLM(n=10)
out = llm("Zoe Kwan is a 20-year old singer and songwriter who has taken Hong Kong’s music scene by storm.")
print(out)
'''

#from langchain.llms.dkube import HuggingFaceHub

#from langchain import HuggingFaceHub
#hf = HuggingFaceHub()

#hf = HuggingFaceHub(repo_id="EleutherAI/gpt-j-6B", huggingfacehub_api_token="my-api-key")
#hf("hi")


