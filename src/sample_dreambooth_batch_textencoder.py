
import json
import torch
import os

from diffusers import StableDiffusionPipeline,UNet2DConditionModel
from diffusers import DiffusionPipeline, DDIMScheduler,PNDMScheduler
from PIL import Image

from PIL import Image
import argparse
from transformers import BlipProcessor, BlipForConditionalGeneration
def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--train",
        type=str,
        default='True',
        choices=['True','False']
        
    )
    parser.add_argument(
        "--edit",
        type=str,
        default='True',
        choices=['True','False']
        
    )
    parser.add_argument(
        "--save",
        type=str,
        default='False',
        choices=['True','False']
        
    )
    parser.add_argument(
        "--interpolation",
        type=str,
        default='vs',
        choices=['vs','vp']
        
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

   
    
    return args
args = parse_args()
# Use a pipeline as a high-level helper
#from transformers import pipeline

#pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

blipmodel='/mnt/bn/editdiffusion/Forgedit/models/blip-image-captioning-base'
#"Salesforce/blip-image-captioning-base"
#os.path.join(MOUNT_OSS_ROOT,'blip-image-captioning-base')

processor = BlipProcessor.from_pretrained(blipmodel)
model = BlipForConditionalGeneration.from_pretrained(blipmodel).to("cuda")
MOUNT_OSS_ROOT = '.'
MOUNT_OSS_ROOT_EXP = os.path.join(MOUNT_OSS_ROOT, 'Experiments')
model_path='stable-diffusion-v1-4/'
diffusion_dir=os.path.join(MOUNT_OSS_ROOT,model_path)

textlr=5e-6

unetlr=5e-6
textsteps=400
unetsteps=400

bsz=4

prompt="A photo of a bird spreading wings."
img_name="bird.jpeg"
img_url=os.path.join(MOUNT_OSS_ROOT,'tedbench/originals',img_name)
init_image = Image.open(img_url).convert("RGB")
if args.save=='True':
    save_sd_path=os.path.join(MOUNT_OSS_ROOT,"dreambooth+bsz={}+textencoder={}+unet={}+tedbench/{}_{}__bsz={}_unetlr={}_textlr={}_{}".format(args.interpolation,bsz,textlr,unetlr,prompt,textsteps,bsz,unetlr,textlr,img_name))
    
    os.makedirs(save_sd_path, exist_ok=True)


has_cuda = torch.cuda.is_available()
device = torch.device('cpu' if not has_cuda else 'cuda')



if args.edit=='True':
    unet_orig = UNet2DConditionModel.from_pretrained(os.path.join(diffusion_dir, 'unet'),
                                                in_channels=4,
                                                low_cpu_mem_usage=False).to(device)




schedule=DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
if args.train=='True':
    pipe = DiffusionPipeline.from_pretrained(
        diffusion_dir,
        safety_checker=None,
        use_auth_token=True,
        custom_pipeline="src/forgedit_stable_diffusion/pipelinedreamboothparallel_bsz=1_textencoder.py",
        scheduler = schedule,
        torch_dtype=torch.float32
    ).to(device)
elif args.edit=='True':
    
    save_sd_path=os.path.join(MOUNT_OSS_ROOT,"dreambooth+bsz={}+textencoder={}+unet={}+tedbench/{}_{}__bsz={}_unetlr={}_textlr={}_{}".format(args.interpolation,bsz,textlr,unetlr,prompt,textsteps,bsz,unetlr,textlr,img_name))
    
    pipe = DiffusionPipeline.from_pretrained(
        save_sd_path,
        safety_checker=None,
        use_auth_token=True,
        custom_pipeline="src/forgedit_stable_diffusion/pipelinedreamboothparallel_bsz=1_textencoder.py",
        scheduler = schedule,
        torch_dtype=torch.float32
    ).to(device)
generator = torch.Generator("cuda").manual_seed(0)
seed = 0              



# unconditional image captioning
inputs = processor(init_image, return_tensors="pt").to("cuda")

out = model.generate(**inputs)
source=processor.decode(out[0], skip_special_tokens=True)
if 'A photo of ' in prompt:
    source='A photo of '+source
print('source=',source)
w,h=init_image.size
init_image = init_image.resize((512, 512))



if args.train=='True':
    res = pipe.train(
        source=source,
        prompt=prompt,
        image=init_image,
        
        embedding_learning_rate=textlr,
        diffusion_model_learning_rate=unetlr,
        
        bsz=bsz,
        text_embedding_optimization_steps= textsteps,
        model_fine_tuning_optimization_steps= unetsteps,
        generator=generator)
    
    
if args.save=='True':
    pipe.save_pretrained(save_sd_path)
if args.edit=='True':
    
    save_edit_path="dreambooth+edit+interpolation={}+bsz={}+textencoder={}+unet={}+tedbench/{}_{}__bsz={}_unetlr={}_textlr={}_{}".format(args.interpolation,bsz,textlr,unetlr,prompt,textsteps,bsz,unetlr,textlr,img_name)
    os.makedirs(os.path.join(MOUNT_OSS_ROOT,save_edit_path), exist_ok=True)
    guide_list=[7.5]
    
    freeze_list=['orig']#['noencoder']#['encoderattn+encoder1']#,'encoderattn','decoderattn']#['encoderkv','decoderkv','memory']#['noencoder']#['orig','decoderattn','encoderattn']#'no','encoder','memory']
    for epoch in range(10):
        for guidance_scale in guide_list:
            for freeze in freeze_list:
                
                
                for i in range(8,17):#change this range for different editing
                
                    num_alpha=i*0.1
                    if args.interpolation=='vp':
                        textlist=[8,9]
                    elif args.interpolation=='vs':
                        textlist=[0]
                    for itextalpha in textlist:
                    
                        textalpha=0.1*itextalpha
                
                        res = pipe(source=source,prompt=prompt,interpolation=args.interpolation,freeze=freeze,unet_orig=unet_orig,alpha=num_alpha, guidance_scale=guidance_scale, num_inference_steps=50,textalpha=textalpha)
                        image=res.images[0]
                        
                        image.save(os.path.join(MOUNT_OSS_ROOT,save_edit_path,str(epoch)+'_'+freeze+'_'+prompt+'_guidance_scale={}_'.format(guidance_scale)+'_'+'textalpha={}_alpha={}'.format(textalpha,num_alpha)+'_'+img_name+'.png'))
                        

del pipe

