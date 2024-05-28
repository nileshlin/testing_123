import logging
import os
from fastapi import FastAPI, Request,HTTPException
import orjson
from timeit import default_timer as timer
from fastapi.middleware.cors import CORSMiddleware
from torch_unit import disable_torch_init
from image_modifiers import load_images
from load_models import LoadVilaImage
import torch
llava_model = LoadVilaImage()
from constants import *
from conversation import conv_templates,SeparatorStyle
import re
from mm_utils import process_images,tokenizer_image_token,KeywordsStoppingCriteria
model_path = '/content/VILA1.5-3b/'
model_name = ''
tokenizer, model, image_processor, context_len = llava_model.load_pretrained_model(model_path)
image_files = ['av.png']
video_file = None

PROMPT = ''

qs = PROMPT

image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
if IMAGE_PLACEHOLDER in qs:
    if model.config.mm_use_im_start_end:
        qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
    else:
        qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
else:
    if DEFAULT_IMAGE_TOKEN not in qs:
        print("no <image> tag found in input. Automatically append one at the beginning of text.")

print("input: ", qs)

if "llama-2" in model_name.lower():
    conv_mode = "llava_llama_2"
elif "v1" in model_name.lower():
    conv_mode = "llava_v1"
elif "mpt" in model_name.lower():
    conv_mode = "mpt"
else:
    conv_mode = "llava_v0"

if conv_mode is not None and conv_mode != conv_mode:
    print(
        "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
            conv_mode, conv_mode, conv_mode
        )
    )
else:
    conv_mode = conv_mode

conv = conv_templates[conv_mode].copy()
conv.append_message(conv.roles[0], qs)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

images = load_images(image_files)
if video_file:
    from llava.mm_utils import opencv_extract_frames
    images = opencv_extract_frames(video_file, frames=6)

images_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)
input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
keywords = [stop_str]
stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

print(images_tensor.shape)
with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        images=[
            images_tensor,
        ],
        do_sample=True,
        temperature=0.2,
        top_p=None,
        num_beams=1,
        max_new_tokens=12,
        use_cache=True,
        stopping_criteria=[stopping_criteria],
    )

outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
outputs = outputs.strip()
if outputs.endswith(stop_str):
    outputs = outputs[: -len(stop_str)]
outputs = outputs.strip()
print(outputs)



