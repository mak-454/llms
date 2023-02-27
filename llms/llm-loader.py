import ray
import json
from ray.serve.drivers import DAGDriver
from ray import serve
from biogpt import BioGpt
from biogpt_large import BioGptLarge
from biogpt_large_pubmedqa import BioGPTLargePubmedQA
from gptjt6b import GptJT6B
from gptj6b import GptJ6B

loaded_models = ["biogpt", "biogpt-large", "biogpt-large-pubmedqa", "gptjt6b", "gptj6b"]

llms_map = {
        "togethercomputer/GPT-JT-6B-v1" : "/gptjt6b",
        "EleutherAI/gpt-j-6B": "/gptj6b",
        "microsoft/biogpt": "/biogpt",
        "microsoft/BioGPT-Large": "/biogpt-large",
        "microsoft/BioGPT-Large-PubMedQA": "/biogpt-large-pubmedqa"
        }


@serve.deployment()
class Probe:
    async def __call__(self, starlette_request):
        request = await starlette_request.body()
        llm = json.loads(request)
        if llm in loaded_models:
            return "deployed"
        else:
            return "not deployed"


@serve.deployment()
class List:
    async def __call__(self, starlette_request):
        data = await starlette_request.body()
        llm = json.loads(data)
        if llm in llms_map.keys():
            return {"name": llm, "status": "deployed", "prefix": llms_map[llm]}
        else:
            return {"name": llm, "status": "not deployed"}        

# Connect to the running Ray cluster.
ray.init(address="auto", namespace="default")
# Bind on 0.0.0.0 to expose the HTTP server on external IPs.
serve.start(detached=True, http_options={"host": "0.0.0.0"})

dag = DAGDriver.bind({
                     "/gptjt6b": GptJT6B.bind(),
                     "/gptj6b": GptJ6B.bind(),
                     "/biogpt": BioGpt.bind(),
                     "/biogpt-large": BioGptLarge.bind(),
                     "/biogpt-large-pubmedqa": BioGPTLargePubmedQA.bind(),
                     "/probe": Probe.bind(),
                     "/llms": List.bind()})

handle = serve.run(dag, host='0.0.0.0')
