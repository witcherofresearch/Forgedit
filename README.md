# Forgedit: Text Guided Image Editing via Learning and Forgetting

This is the official implementation of my paper forgedit: text guided image editing via learning and forgetting.

## Abstract 

Text guided image editing on real images given only the image itself and the target text prompt as inputs, is a very general and challenging problem, which requires the editing model to  reason by itself which part of the image should be edited, to preserve the characteristics of original image, and also to perform complicated non-rigid editing.   Previous fine-tuning based solutions are time-consuming  and vulnerable to overfitting, limiting their editing capabilities. To tackle these issues, we design a novel text guided image editing method, Forgedit. First, we propose a novel fine-tuning framework which learns to reconstruct the given image in less than one minute by vision language joint learning. Then we introduce vector subtraction and vector projection to explore the proper text embedding for editing. We also find a general property of UNet structures in Diffusion Models and inspired by such a founding, we design a forgetting strategy to diminish the fatal overfitting issues and significantly boost the editing ability of Diffusion Models. Our method, Forgedit, implemented with Stable Diffusion, achieves new state-of-the-art results on the challenging text guided image editing benchmark TEdBench,  surpassing the previous SOTA method Imagic with Imagen, in terms of both CLIP score and LPIPS score.


## Acknowledgement

This code is based on Diffusers implemented [Imagic](https://github.com/huggingface/diffusers/blob/main/examples/community/imagic_stable_diffusion.py)

## Installation

Please make sure you can run [Diffusers](https://github.com/huggingface/diffusers/), 
better install xformer too in order to speedup training and sampling.

## Forgedit with Stable Diffusion 1.4

Please first download Stable Diffusion 1.4 and the TEdBench text guided image editing benchmark.

In this code release, Forgedit and DreamBoothForgedit are implemented. 

### Forgedit

The saving and loading functions of vanilla Forgedit is not implemented 
yet and will be released in the next version. To edit the image with vanilla Forgedit and
interpolate text embeddings with  vector subtraction, 

```
accelerate launch src/sample_forgedit_batch_textencoder.py --interpolation=vs
```

Forgetting strategies are implemented in src/forgedit_stable_diffusion/pipelineattentionparallel_bsz=1.py,
which can be used in the freeze_list in sample_forgedit_batch_textencoder.py




### DreamBoothForgedit

To fine-tune, save and edit with DreamBoothForgedit with vector projection,


```
accelerate launch src/sample_dreambooth_batch_textencoder.py --save=True --interpolation=vp
```


To edit with saved editing models, 


```
accelerate launch src/sample_dreambooth_batch_textencoder.py --train=False --interpolation=vp
```

Forgetting strategies are implemented in src/forgedit_stable_diffusion/pipelinedreamboothparallel_bsz=1_textencoder.py,
which can be used in the freeze_list in sample_dreambooth_batch_textencoder.py


## TEdBench

The complete editing results of vanilla Forgedit on TEdBench can be found in the [tedbench repository](https://github.com/witcherofresearch/tedbench). Please note that these editing results are carelessly manually selected 
and are not final thus could be improved in the next version. The complete results of DreamBooth Forgedit are not provided in this release. 

## Citation

The paper will soon be released on arxiv.