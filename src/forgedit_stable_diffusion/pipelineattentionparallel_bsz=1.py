
import inspect
import warnings
from typing import List, Optional, Union
import itertools

import numpy as np
import torch
import torch.nn.functional as F
import copy

import PIL
from accelerate import Accelerator
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from diffusers.utils import deprecate, logging
from torch import linalg as LA

from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer,BertTokenizer, BertModel


if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }
# ------------------------------------------------------------------------------

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL_INTERPOLATION["lanczos"])
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


class ForgeditStableDiffusionPipeline(DiffusionPipeline):
    

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.unet.set_use_memory_efficient_attention_xformers(True)

    def enable_attention_slicing(self, slice_size: Optional[Union[str, int]] = "auto"):
        r"""
        Enable sliced attention computation.
        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.
        Args:
            slice_size (`str` or `int`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                a number is provided, uses as many slices as `attention_head_dim // slice_size`. In this case,
                `attention_head_dim` must be a multiple of `slice_size`.
        """
        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = self.unet.config.attention_head_dim // 2
        self.unet.set_attention_slice(slice_size)

    def disable_attention_slicing(self):
        r"""
        Disable sliced attention computation. If `enable_attention_slicing` was previously invoked, this method will go
        back to computing attention in one step.
        """
        # set slice_size = `None` to disable `attention slicing`
        self.enable_attention_slicing(None)

    def train(
        self,
        source:Union[str, List[str]],
        prompt: Union[str, List[str]],
        image: Union[torch.FloatTensor, PIL.Image.Image],
        unet_orig=None,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        bsz=1,
        generator: Optional[torch.Generator] = None,
        embedding_learning_rate: float = 1e-3,#0.001,
        diffusion_model_learning_rate: float = 2e-5,
        memory_learning_rate=2e-5,
        text_embedding_optimization_steps: int = 500,
        model_fine_tuning_optimization_steps: int = 1000,
        **kwargs,
    ):
        
        
        
        
        
        self.prompt=prompt

        accelerator = Accelerator(
            gradient_accumulation_steps=1,
            mixed_precision="fp16",
        )

        if "torch_device" in kwargs:
            device = kwargs.pop("torch_device")
            warnings.warn(
                "`torch_device` is deprecated as an input argument to `__call__` and will be removed in v0.3.0."
                " Consider using `pipe.to(torch_device)` instead."
            )

            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.to(device)

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.eval()
        self.vae.eval()
        self.text_encoder.eval()
        

        if accelerator.is_main_process:
            accelerator.init_trackers(
                "imagic",
                config={
                    "embedding_learning_rate": embedding_learning_rate,
                    "text_embedding_optimization_steps": text_embedding_optimization_steps,
                },
            )

        # get text embeddings for prompt
        text_input = self.tokenizer(
            
            source,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        text_embeddings =self.text_encoder(text_input.input_ids.to(self.device))[0]
        text_embeddings = text_embeddings.detach()
        
        text_embeddings = torch.nn.Parameter(
            text_embeddings, requires_grad=True
        )
        
        
        
        
        
        b,n,c=text_embeddings.shape
        
        
        text_embeddings_orig = self.tokenizer(
            
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings_orig= self.text_encoder(text_embeddings_orig.input_ids.to(self.device))[0]
        text_embeddings_orig = text_embeddings_orig.detach()
        
        
        
        
        
        
        
        self.unet.train()
        params_to_optimize =[]
        
        for name, params in self.unet.named_parameters():
            
            params.requires_grad=False
            
            
            
            #encoderdecoder
            if   ('up_blocks' in name ) and not 'up_blocks.0' in name:# and not 'attn2.to_v' in name and not'attn2.to_k' in name :#'down_blocks.3' in name:
                params.requires_grad = True
            if   ('down_blocks' in name ) and not 'down_blocks.0' in name:# and not 'attn2.to_v' in name and not'attn2.to_k' in name:#'down_blocks.3' in name:
                params.requires_grad = True
            
            
            
            if params.requires_grad==True:
                params_to_optimize.append(params)
                print(name, " with graidient")
        
        
        optimizer = torch.optim.Adam([
            {'params':params_to_optimize,  
            'lr':diffusion_model_learning_rate,},
            {'params':text_embeddings,'lr':embedding_learning_rate},
            
        ])
        progress_bar = tqdm(range(text_embedding_optimization_steps//bsz), disable=not accelerator.is_local_main_process)
        params_to_accumulate = (
            itertools.chain(text_embeddings,params_to_optimize)#[x[1] for x in self.unet.named_parameters() if ('attn2.to_k' in x[0] or 'attn2.to_v' in x[0]   or 'up_blocks.2' in x[0]   )] ) 
        )
       
        if isinstance(image, PIL.Image.Image):
            image = preprocess(image)
        
        latents_dtype = text_embeddings.dtype
        image = image.to(device=self.device, dtype=latents_dtype)
        init_latent_image_dist = self.vae.encode(image).latent_dist
        image_latents = init_latent_image_dist.sample(generator=generator)
        image_latents = 0.18215 * image_latents
        
        global_step = 0
        self.unet_list=[]
        self.text_list=[]

        logger.info("First optimizing the text embedding to better reconstruct the init image")
        for i in range(text_embedding_optimization_steps//bsz):
            with accelerator.accumulate(params_to_accumulate):#text_embeddings):
                # Sample noise that we'll add to the latents
                image_latents_batch=image_latents.repeat(bsz,1,1,1)
                noise = torch.randn(image_latents_batch.shape).to(image_latents.device)
                text_embeddings_batch=text_embeddings.repeat(bsz,1,1)#*attention.repeat(bsz,1,1)
                timesteps = torch.randint(1000, (bsz,), device=image_latents.device)
                
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = self.scheduler.add_noise(image_latents_batch, noise, timesteps)

                # Predict the noise residual
                noise_pred = self.unet(noisy_latents, timesteps, text_embeddings_batch).sample

                loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
                #print('loss {}={}'.format(i,loss.item()))
                if loss.item()<0.03 and i*bsz>350:
                    print('final loss is ',loss.item())
                    print('step is {}'.format(i))
                    break
                accelerator.backward(loss)

                optimizer.step()
                optimizer.zero_grad()
                

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {"loss": loss.detach().item()}  # , "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            
        accelerator.wait_for_everyone()
        
        text_embeddings.requires_grad_(False)
        
        for name, params in self.unet.named_parameters():
            params.requires_grad=False

        
        
        self.unet.eval()
        
        
        self.text_embeddings_orig = text_embeddings_orig
        self.text_embeddings = text_embeddings
        
        
        torch.cuda.empty_cache()
    
    @torch.no_grad()
    def __call__(
        self,
        unet_orig=None,
        prompt='',
        alpha: float = 1.2,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        ForgetDegree=0.0,
        interpolation='vp',
        freeze='',
        textalpha=0,
        **kwargs,
    ):
        
        self.unet.eval()
        self.unet_copy=copy.deepcopy(self.unet)
        
        
        
        for u,u_orig in zip(self.unet_copy.named_parameters(),unet_orig.named_parameters()):
            name,params=u
            name_orig,params_orig=u_orig
            
            params.requires_grad = False
            
            if freeze=='encoderkv':
                if 'down_blocks' in name and not ( 'attn2.to_v' in name or 'attn2.to_k' in name): 
                    params.data=ForgetDegree*params.data+(1-ForgetDegree)*params_orig.data
            elif freeze=='noencoder':
                if 'down_blocks' in name:
                    params.data=ForgetDegree*params.data+(1-ForgetDegree)*params_orig.data
            
            elif freeze=='encoderattn':
                if 'down_blocks' in name and not ( 'attn' in name ):
                    
                    params.data=ForgetDegree*params.data+(1-ForgetDegree)*params_orig.data
            elif freeze=='encoderattn+encoder1':
                if 'down_blocks' in name and not ( 'attn' in name  or 'down_blocks.1' in name):#
                    
                    params.data=ForgetDegree*params.data+(1-ForgetDegree)*params_orig.data
            elif freeze=='decoderkv':
                if ( 'up_blocks' in name) and not   ('attn2.to_k' in name or 'attn2.to_v' in name ) :
                    
                    params.data=ForgetDegree*params.data+(1-ForgetDegree)*params_orig.data
            elif freeze=='decoderattn':
                if ( 'up_blocks' in name) and not ( 'attn' in name):
                    
                    params.data=ForgetDegree*params.data+(1-ForgetDegree)*params_orig.data
            elif freeze=='decoderattn+decoder2':
                if ( 'up_blocks' in name) and not ( 'attn' in name or 'up_blocks.2' in name ):
                    
                    params.data=ForgetDegree*params.data+(1-ForgetDegree)*params_orig.data
            
            
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
        if self.text_embeddings is None:
            raise ValueError("Please run the pipe.train() before trying to generate an image.")
        
        
        text_embeddings=self.text_embeddings
        prompt_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_embeddings = self.text_encoder(prompt_input.input_ids.to(self.device))[0]
        
        #prompt_embeddings=self.text_embeddings_orig
        
        if interpolation=='vp':
            normalizetext=torch.nn.functional.normalize(text_embeddings,dim=2)
            b,n,c=normalizetext.shape
            print(b,n,c)
            normtext=normalizetext.view(n,c,1)
            viewprompt=prompt_embeddings.view(n,1,c)
            projtext=torch.matmul(viewprompt,normtext)
            projtext=projtext*normtext
            projtext=projtext.view(1,n,c)
            projedit=prompt_embeddings-projtext
            
        
            
            text_embeddings =alpha*projedit+textalpha*text_embeddings
        elif interpolation=='vs':
            text_embeddings = alpha * prompt_embeddings + (1 - alpha) * text_embeddings
        

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens = [""]
            max_length = self.tokenizer.model_max_length
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.view(1, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
            
        # get the initial random noise unless the user supplied it

        # Unlike in other pipelines, latents need to be generated in the target device
        # for 1-to-1 results reproducibility with the CompVis implementation.
        # However this currently doesn't work in `mps`.
        latents_shape = (1, self.unet.in_channels, height // 8, width // 8)
        latents_dtype = text_embeddings.dtype
        
        if self.device.type == "mps":
            # randn does not exist on mps
            latents = torch.randn(latents_shape, generator=generator, device="cpu", dtype=latents_dtype).to(
                self.device
            )
        else:
            latents = torch.randn(latents_shape, generator=generator, device=self.device, dtype=latents_dtype)
       
        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        # Some schedulers like PNDM have timesteps as arrays
        # It's more optimized to move all timesteps to correct device beforehand
        timesteps_tensor = self.scheduler.timesteps.to(self.device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
            
        
        #for unet in self.unet_list:
        for i, t in enumerate(self.progress_bar(timesteps_tensor)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            print(t)
            if True:#i<thresh_t:
                noise_pred = self.unet_copy(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            else:
                noise_pred = unet_orig(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            
            

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(
                self.device
            )
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(text_embeddings.dtype)
            )
        else:
            has_nsfw_concept = None

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)