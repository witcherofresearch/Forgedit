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
    parser = argparse.ArgumentParser(description="vanilla Forgedit")
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
        "--loadfrom",
        type=str,
        default='',
        
        
    )
    parser.add_argument(
        "--interpolation",
        type=str,
        default='vs',
        choices=['vs','vp']
        
    )
    parser.add_argument(
        "--targetw",
        type=int,
        default=512,
        
        
    )
    parser.add_argument(
        "--targeth",
        type=int,
        default=512,
        
        
    )
    parser.add_argument(
        "--gammastart",
        type=int,
        default=0,
        
        
    )
    parser.add_argument(
        "--gammaend",
        type=int,
        default=17,
        
        
    )
    parser.add_argument(
        "--numtest",
        type=int,
        default=4,
        
        
    )
    parser.add_argument(
        "--forget",
        type=str,
        default='donotforget',
        
        
    )
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

   
    
    return args
args = parse_args()
MOUNT_OSS_ROOT = '.'

diffusion_dir='/mnt/bn/editdiffusion/Forgedit/models/models--SG161222--Realistic_Vision_V6.0_B1_noVAE/snapshots/7f177697718f088f243fd357263b9f0cb22d0cac'
#'SG161222/Realistic_Vision_V6.0_B1_noVAE'#'stable-diffusion-v1-4/'

textlr=1e-3
unetlr=6e-5

textsteps=400
unetsteps=400

bsz=10



prompt='a man and a woman at new york, skyscrapers'


img_url='./test.png'

img_name=img_url.split('/')[-1]
init_image = Image.open(img_url).convert("RGB")
if args.save=='True':
    save_sd_path=os.path.join(MOUNT_OSS_ROOT,"vanillaforgedit/img={}_textsteps={}_bsz={}_unetlr={}_textlr={}".format(img_name,textsteps,bsz,unetlr,textlr))
    
    os.makedirs(save_sd_path, exist_ok=True)

has_cuda = torch.cuda.is_available()
device = torch.device('cpu' if not has_cuda else 'cuda')



if args.edit=='True':
    
    unet_orig = UNet2DConditionModel.from_pretrained(os.path.join(diffusion_dir, 'unet'),
                                                in_channels=4,
                                                low_cpu_mem_usage=False).to(device)

#blipmodel="Salesforce/blip-image-captioning-base"
blipmodel='/mnt/bn/editdiffusion/Forgedit/models/blip-image-captioning-base'
processor = BlipProcessor.from_pretrained(blipmodel)
model = BlipForConditionalGeneration.from_pretrained(blipmodel).to("cuda")



schedule=DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
if args.train=='True':
    pipe = DiffusionPipeline.from_pretrained(
        diffusion_dir,
        safety_checker=None,
        use_auth_token=True,
        custom_pipeline="src/forgedit_stable_diffusion/pipelineattentionparallel_bsz=1.py",
        scheduler = schedule,
        torch_dtype=torch.float32
    ).to(device)
elif args.edit=='True':
    save_sd_path=args.loadfrom

    
    pipe = DiffusionPipeline.from_pretrained(
        save_sd_path,
        safety_checker=None,
        use_auth_token=True,
        custom_pipeline="src/forgedit_stable_diffusion/pipelineattentionparallel_bsz=1.py",
        scheduler = schedule,
        torch_dtype=torch.float32
    ).to(device)
    pipe.text_embeddings=torch.load(os.path.join(save_sd_path,'src+text+embeddding.pt')).to(device)
generator = torch.Generator("cuda").manual_seed(0)
seed = 0              



# unconditional image captioning
inputs = processor(init_image, return_tensors="pt").to("cuda")

out = model.generate(**inputs)
source=processor.decode(out[0], skip_special_tokens=True)
if 'A photo of ' in prompt:
    source='A photo of '+source
print('source=',source)

w,h=args.targetw,args.targeth
init_image = init_image.resize((w, h))



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
    torch.save(pipe.text_embeddings.cpu(),os.path.join(save_sd_path,'src+text+embeddding.pt'))
    pipe.save_pretrained(save_sd_path)

if args.edit=='True':
    
    save_edit_path="forgedit+edit+interpolation={}+bsz={}+textencoder={}+unet={}+tedbench/{}_{}__bsz={}_unetlr={}_textlr={}_{}".format(args.interpolation,bsz,textlr,unetlr,prompt,textsteps,bsz,unetlr,textlr,img_name)
    os.makedirs(os.path.join(MOUNT_OSS_ROOT,save_edit_path), exist_ok=True)
    guide_list=[7.5]
    
    freeze_list=[args.forget]
    #['donotforget'] refers to do not forget any learned parameters
    #['encoderattn'] refers to forget all parameters of UNet encoder other than attention modules
    #['decoderattn'] refers to forget all parameters of UNet decoder other than attention modules
    # for more options and implementations, please check "src/forgedit_stable_diffusion/pipelineattentionparallel_bsz=1.py"
    for epoch in range(args.numtest):
        for guidance_scale in guide_list:
            for freeze in freeze_list:
                for i in range(args.gammastart,args.gammaend):#this range should be changed according to different interpolation and forgetting strategies, maybe change this range to [10,17] for vp, 
                    #gamma refers to gamma in vector subtraction and beta in vector projection
                    num_alpha=i*0.1
                    if args.interpolation=='vp':
                        textlist=[8,9,11]
                    else:
                        textlist=[0]
                    for itextalpha in textlist:
                    
                        textalpha=0.1*itextalpha
                
                        res = pipe(source=source,prompt=prompt,interpolation=args.interpolation,freeze=freeze,unet_orig=unet_orig,alpha=num_alpha, guidance_scale=guidance_scale, num_inference_steps=50,textalpha=textalpha,height=h,width=w)
                        image=res.images[0]
                        
                        image.save(os.path.join(MOUNT_OSS_ROOT,save_edit_path,str(epoch)+'_'+freeze+'_'+prompt+'_guidance_scale={}_'.format(guidance_scale)+'_'+'textalpha={}_alpha={}'.format(textalpha,num_alpha)+'_'+img_name+'.png'))
                        

del pipe
