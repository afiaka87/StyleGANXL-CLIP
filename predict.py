# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import sys
# sys.path.append('./CLIP') # we use `clip-anytorch` instead to avoid sys.path import
sys.path.append('./stylegan_xl')

import cog
from cog import BasePredictor, Input

import typing
import tempfile
import time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import clip
import unicodedata
import re
from tqdm import tqdm
from torchvision.transforms import Compose
from einops import rearrange
import legacy
import dnnlib

def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')

def norm1(prompt):
    "Normalize to the unit sphere."
    return prompt / prompt.square().sum(dim=-1,keepdim=True).sqrt()

def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

def prompts_dist_loss(x, targets, loss):
    if len(targets) == 1: # Keeps consitent results vs previous method for single objective guidance 
      return loss(x, targets[0])
    distances = [loss(x, target) for target in targets]
    return torch.stack(distances, dim=-1).sum(dim=-1)  

class MakeCutouts(torch.nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)

class CLIP(object):
    def __init__(
        self,
        model_name,
        device
    ):
        clip_model = model_name
        self.device = device
        self.model, _ = clip.load(clip_model)
        self.model = self.model.requires_grad_(False)
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

    @torch.no_grad()
    def embed_text(self, prompt):
        return norm1(self.model.encode_text(clip.tokenize(prompt, truncate=True).to(self.device)).float())

    def embed_cutout(self, image):
        return norm1(self.model.encode_image(self.normalize(image)))
  
sgxl_named_checkpoints = { 
    "Imagenet512": "imagenet512.pkl",
    # "Imagenet128": "imagenet128.pkl",
    # "Pokemon": "pokemon256.pkl",
    # "FFHQ": "ffhq256.pkl" 
}


class Predictor(BasePredictor):
    def setup_stylegan_xl_cond(self, model_name, device):
        with open(sgxl_named_checkpoints[model_name], 'rb') as f:
            self.G = legacy.load_network_pkl(f)['G_ema'].to(self.device)

        zs = torch.randn([10000, self.G.mapping.z_dim], device=device)
        one_hot_class = torch.zeros(1000)
        initial_class = torch.randint(0,1000, (1,))[0]
        one_hot_class[initial_class]=1
        one_hot_class = one_hot_class.repeat((10000, 1))
        cs = one_hot_class.to(device)
        w_stds = self.G.mapping(zs, cs)
        w_stds=w_stds.std(0)[0]
        c = torch.zeros((1000)) #just to pick a closer initial image
        c[initial_class]=1
        c = c.repeat(self.initial_batch, 1)
        c=c.to(device)

    def setup_stylegan_xl(self, model_name='Imagenet512'):
        with open(sgxl_named_checkpoints[model_name], 'rb') as f:
            self.G = legacy.load_network_pkl(f)['G_ema'].to(self.device)

        zs = torch.randn([10000, self.G.mapping.z_dim], device=self.device)
        cs = torch.zeros([10000, self.G.mapping.c_dim], device=self.device)
        for i in range(cs.shape[0]):
            cs[i,i//10]=1
        self.w_stds = self.G.mapping(zs, cs)
        self.w_stds = self.w_stds.reshape(10, 1000, self.G.num_ws, -1)
        self.w_stds = self.w_stds.std(0).mean(0)[0]
        self.w_all_classes_avg = self.G.mapping.w_avg.mean(0)

        # Calculating a gradient with the model will force the 'lrelu_filtered' torch extension to compile.
        # This takes some time, so it's best to do it once in setup.
        with torch.no_grad():
            a = torch.randn([self.initial_batch, 512], device=self.device)*0.6 + self.w_stds*0.4
            q = ((a-self.w_all_classes_avg)/self.w_stds)
            images = self.G.synthesis((q * self.w_stds + self.w_all_classes_avg).unsqueeze(1).repeat([1, self.G.num_ws, 1]))
            embeds = self.embed_image(images.add(1).div(2))
            targets = [self.clip_model.embed_text("test")]
        prompts_dist_loss(embeds, targets, spherical_dist_loss).mean(0)

    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = torch.device('cuda')
        # self.make_cutouts = MakeCutouts(224, 32, 0.5)
        self.clip_model = CLIP(model_name="ViT-L/14@336px", device=self.device)
        self.make_cutouts = MakeCutouts(336, 12, 0.5)
        self.initial_batch = 4 #actually that will be multiplied by initial_image_steps

        # Checkpoints can be switched by the user, but we load the default here. This also forces the model to be compiled.
        self.setup_stylegan_xl(model_name="Imagenet512")


    def embed_image(self, image):
        n = image.shape[0]
        cutouts = self.make_cutouts(image)
        embeds = self.clip_model.embed_cutout(cutouts)
        embeds = rearrange(embeds, '(cc n) c -> cc n c', n=n)
        return embeds

    def embed_url(self, _path):
        image = Image.open(_path).convert('RGB')
        return self.embed_image(TF.to_tensor(image).to(self.device).unsqueeze(0)).mean(0).squeeze(0)
    

    def predict(
        self,
        texts: str = cog.Input(
            description="""Enter here a prompt to guide the image generation.  You can enter more than one prompt separated with |,
            which will cause the guidance to focus on the different prompts at the same time, allowing to mix and play with the generation process.""", 
            default="an image of a cat|digital illustration"),
        steps: int = cog.Input(
            description="Number of optimization steps. The more steps, the longer it will try to generate an image relevant to the prompt.",
            default=100),
        seed: int = cog.Input(
            description="Determines the randomness seed. Using the same seed and prompt should give you similar results at every run.",
            default=-1
        ),
        lr: float = cog.Input(
            description="Learning rate for AdamW optimizer.", 
            default=0.05),
    ) -> typing.Iterator[cog.Path]:
        torch.manual_seed(seed)
        tf = Compose([
            # Resize(224),
            lambda x: torch.clamp((x+1)/2,min=0,max=1),
        ])
        prefix = cog.Path(tempfile.mkdtemp())

        self.initial_batch=4 #actually that will be multiplied by initial_image_steps
        initial_image_steps=8

        if seed == -1:
            seed = np.random.randint(0,9e9)
            print(f"Your random seed is: {seed}")

        texts = [frase.strip() for frase in texts.split("|") if frase]
        print(f"You entered {len(texts)} prompt/s: {texts}")

        targets = [self.clip_model.embed_text(text) for text in texts]


        with torch.no_grad():
            qs = []
            losses = []
            for _ in range(initial_image_steps):
                a = torch.randn([self.initial_batch, 512], device=self.device)*0.6 + self.w_stds*0.4
                q = ((a-self.w_all_classes_avg)/self.w_stds)
                images = self.G.synthesis((q * self.w_stds + self.w_all_classes_avg).unsqueeze(1).repeat([1, self.G.num_ws, 1]))
                embeds = self.embed_image(images.add(1).div(2))
                loss = prompts_dist_loss(embeds, targets, spherical_dist_loss).mean(0)
                i = torch.argmin(loss)
                qs.append(q[i])
                losses.append(loss[i])
            qs = torch.stack(qs)
            losses = torch.stack(losses)
            i = torch.argmin(losses)
            q = qs[i].unsqueeze(0)
        q.requires_grad_(True)


        # Sampling loop
        print(f"Sampling {steps} steps.")

        q_ema = q
        opt = torch.optim.AdamW([q], lr=lr, betas=(0.5,0.999))
        # loop = tqdm(range(steps))
        loop = range(steps)
        for i in loop:
            opt.zero_grad()
            image = self.G.synthesis((q * self.w_stds + self.w_all_classes_avg).unsqueeze(1).repeat([1, self.G.num_ws, 1]), noise_mode='const')
            embed = self.embed_image(image.add(1).div(2))
            loss = prompts_dist_loss(embed, targets, spherical_dist_loss).mean()
            loss.backward()
            opt.step()
            # loop.set_postfix(loss=loss.item(), q_magnitude=q.std().item())
            print(f"Step {i}/{steps}: loss={loss.item()}, q_magnitude={q.std().item()}")

            q_ema = q_ema * 0.9 + q * 0.1
            image = self.G.synthesis((q_ema * self.w_stds + self.w_all_classes_avg).unsqueeze(1).repeat([1, self.G.num_ws, 1]), noise_mode='const')

            intermediate_output_filename = prefix.joinpath(f"{i}.png")
            TF.to_pil_image(tf(image)[0]).save(intermediate_output_filename)
            yield intermediate_output_filename