import ray
import json
from ray.serve.drivers import DAGDriver
from ray import serve
from biogpt import BioGpt
from biogpt_large import BioGptLarge
from biogpt_large_pubmedqa import BioGPTLargePubmedQA
from gptjt6b import GptJT6B

loaded_models = ["biogpt", "biogpt-large", "biogpt-large-pubmedqa", "gptjt6b"]

@serve.deployment()
class Probe:
    async def __call__(self, starlette_request):
        request = await starlette_request.body()
        llm = json.loads(request)
        if llm in loaded_models:
            return "deployed"
        else:
            return "not deployed"

# Connect to the running Ray cluster.
ray.init(address="auto", namespace="default")
# Bind on 0.0.0.0 to expose the HTTP server on external IPs.
serve.start(detached=True, http_options={"host": "0.0.0.0"})

dag = DAGDriver.bind({"/biogpt": BioGpt.bind(),
                     "/biogpt-large": BioGptLarge.bind(),
                     "/biogpt-large-pubmedqa": BioGPTLargePubmedQA.bind(),
                     "/gptjt6b": GptJT6B.bind(),
                     "/probe": Probe.bind()})

handle = serve.run(dag, host='0.0.0.0')
