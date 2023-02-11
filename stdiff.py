from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
import os
from typing import Dict
from PIL import Image
from numpy import asarray

from ray import serve

model_id = "stabilityai/stable-diffusion-2"

@serve.deployment(route_prefix="/stdiffusion")
class PyTorchModel:
    def __init__(self):
        modelname = os.getenv("MODEL_NAME", "stdiff2")
        num_of_gpus = torch.cuda.device_count()
        print(num_of_gpus)
        
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
        self.pipe = self.pipe.to("cuda")

    async def __call__(self, starlette_request):
      request = await starlette_request.body()
      prompt = request["prompt"]
      
      image = self.pipe(prompt).images[0]
      # convert image to numpy array
      data = asarray(image)
      image.save("result.png")
      return {"result": data.tolist()}
stdiffmodel = PyTorchModel.bind()
