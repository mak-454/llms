#https://huggingface.co/sentence-transformers/all-mpnet-base-v2
from sentence_transformers import SentenceTransformer
import torch
import json

from ray import serve

@serve.deployment(name="embedding", route_prefix="/mpnetbasev2", health_check_timeout_s=600)
class MpnetBaseV2:
    def __init__(self):
        #self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.model = SentenceTransformer('/mnt/llm-cache/sentence-transformers_all-mpnet-base-v2')

    async def __call__(self, starlette_request):
        request = await starlette_request.body()
        sentences = json.loads(request)
        embeddings = self.model.encode(sentences)
        return embeddings

MpnetBaseV2.deploy()
