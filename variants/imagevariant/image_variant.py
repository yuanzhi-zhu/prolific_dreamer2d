import PIL
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

url = "https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg"
def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image
image = download_image(url)

prompt = "turn him into cyborg"
import pdb; pdb.set_trace()
images = pipe(prompt, image=image, num_inference_steps=10, image_guidance_scale=1).images
images[0]

device = "cuda:0"

prompt_embeds = pipe._encode_prompt(prompt,device,num_images_per_prompt=1,do_classifier_free_guidance=True,negative_prompt=None,)

from diffusers import StableDiffusionImageVariationPipeline
from PIL import Image
from torchvision import transforms

sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
  "lambdalabs/sd-image-variations-diffusers",
  revision="v2.0",
  )
sd_pipe = sd_pipe.to(device)
guidance_scale = 3
num_inference_steps=50
im = Image.open("/cluster/home/jinliang/work/ckpts_yuazhu/prolific_dream2d/figs/final_image_a_photograph_of_an_astronaut_riding_a_horse.png")
tform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(
        (224, 224),
        interpolation=transforms.InterpolationMode.BICUBIC,
        antialias=False,
        ),
    transforms.Normalize(
      [0.48145466, 0.4578275, 0.40821073],
      [0.26862954, 0.26130258, 0.27577711]),
])
inp = tform(im).to(device).unsqueeze(0)

import pdb;pdb.set_trace()

out = sd_pipe(inp, guidance_scale=guidance_scale)
out["images"][0].save("result.jpg")

emb = sd_pipe._encode_image(inp, device, num_images_per_prompt=1, do_classifier_free_guidance=True)

import pdb;pdb.set_trace()

import torch
from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import save_image
from transformers import CLIPVisionModelWithProjection
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import DDIMScheduler
### use an img_variant model instead of lora
dtype = torch.float32
# args.save_phi_model = False
### https://huggingface.co/lambdalabs/sd-image-variations-diffusers
phi_model_path = 'lambdalabs/sd-image-variations-diffusers'
image_encoder = CLIPVisionModelWithProjection.from_pretrained(phi_model_path, revision="v2.0", subfolder="image_encoder", torch_dtype=dtype).to(device)
vae_phi = AutoencoderKL.from_pretrained(phi_model_path, revision="v2.0", subfolder="vae", torch_dtype=dtype).to(device)
unet_phi = UNet2DConditionModel.from_pretrained(phi_model_path, revision="v2.0", subfolder="unet", torch_dtype=dtype).to(device)
image_encoder.requires_grad_(False)
vae_phi.requires_grad_(False)
unet_phi.requires_grad_(False)
# tform = transforms.Compose([
#     transforms.Resize(
#         (224, 224),
#         interpolation=transforms.InterpolationMode.BICUBIC,
#         antialias=False,
#         ),
#     transforms.Normalize(
#     [0.48145466, 0.4578275, 0.40821073],
#     [0.26862954, 0.26130258, 0.27577711]),
# ])


img_emb = image_encoder(inp).image_embeds
img_emb = img_emb.unsqueeze(1)
negative_prompt_embeds = torch.zeros_like(img_emb)
# CFG embedding of the image, but still in the name of text_embeddings
text_embeddings_img_variant = torch.cat([negative_prompt_embeds, img_emb])  

scheduler = DDIMScheduler.from_pretrained(phi_model_path, subfolder="scheduler", torch_dtype=dtype)
# scheduler = KarrasDiffusionSchedulers.from_pretrained(phi_model_path, subfolder="scheduler", torch_dtype=dtype)
scheduler.set_timesteps(num_inference_steps)
latents = torch.randn((1, 4, 64, 64)).to(device)

import pdb;pdb.set_trace()

step = 0
for t in tqdm(scheduler.timesteps):
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, t)
    # predict the noise residual
    with torch.no_grad():
        noise_pred = unet_phi(latent_model_input, t, encoder_hidden_states=text_embeddings_img_variant).sample
    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    # compute the previous noisy sample x_t -> x_t-1
    latents = scheduler.step(noise_pred, t, latents).prev_sample
    ######## Evaluation and log metric #########
# 7. Denoising loop
latents = 1 / 0.18215 * latents.clone()
with torch.no_grad():
    image = vae_phi.decode(latents).sample.to(torch.float32)
    save_image((image/2+0.5).clamp(0, 1), f'final_image_test0.png')


import pdb;pdb.set_trace()


# extra_step_kwargs = sd_pipe.prepare_extra_step_kwargs(generator= None, eta=0)
# num_warmup_steps = len(scheduler.timesteps) - timesteps * scheduler.order
for i, t in enumerate(sd_pipe.scheduler.timesteps):
    # expand the latents if we are doing classifier free guidance
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = sd_pipe.scheduler.scale_model_input(latent_model_input, t)

    # predict the noise residual
    noise_pred = unet_phi(latent_model_input, t, encoder_hidden_states=text_embeddings_img_variant).sample

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # compute the previous noisy sample x_t -> x_t-1
    # latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
    latents = sd_pipe.scheduler.step(noise_pred, t, latents).prev_sample
latents = 1 / 0.18215 * latents.clone()
with torch.no_grad():
    image = vae_phi.decode(latents).sample.to(torch.float32)
    save_image((image/2+0.5).clamp(0, 1), f'final_image_test1.png')

import pdb;pdb.set_trace()
# tmp_latents = 1 / 0.18215 * latents_vsd.clone().detach()
# image_vsd = vae.decode(tmp_latents).sample.to(torch.float32)
# # image_vsd = F.interpolate(image_vsd, (224, 224), mode="bicubic", align_corners=False)
# # image_vsd = (image_vsd + 1.) / 2.
# image_vsd = tform(image_vsd).to(device)
# import pdb;pdb.set_trace()
# img_emb = image_encoder.get_image_features(img_emb)
# img_emb = image_encoder(pixel_values=image_vsd)
# img_emb = img_emb.last_hidden_state
# img_emb = F.pad(img_emb, [0,0, 0,77-img_emb.shape[1], 0,0])
# img_emb = img_emb.repeat(args.particle_num_vsd, 1, 1)
# text_embeddings_vsd = torch.cat([uncond_embeddings[:args.particle_num_vsd], img_emb])