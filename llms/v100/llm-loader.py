import ray
import json
from ray.serve.drivers import DAGDriver
from ray import serve
from gptjt6b import GptJT6Bv100

loaded_models = ["gptjt6b"]

llms_map = {
        "togethercomputer/GPT-JT-6B-v1" : "/gptjt6b",
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
                     "/gptjt6b": GptJT6Bv100.bind(),
                     "/probe": Probe.bind(),
                     "/llms": List.bind()})

handle = serve.run(dag, host='0.0.0.0')
