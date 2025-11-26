import argparse
import os
import random
from typing import List, Tuple
import json

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image

from transformers import StoppingCriteriaList

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *



def parse_args():
    parser = argparse.ArgumentParser(description="CytoGPT Local Script")
    parser.add_argument("--cfg-path",default='eval_configs/Cyto_minigpt4_eval1.yaml', help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--image_path", default='', help="specify the image path for evaluation.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def initialize_model(args, cfg):
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(f'cuda:{args.gpu_id}')

    conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0, 'pretrain_llama2': CONV_VISION_LLama2}
    CONV_VISION = conv_dict[model_config.model_type]

    vis_processor_cfg = cfg.datasets_cfg.cyto_vqa.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    stop_words_ids = [[835], [2277, 29937]]
    stop_words_ids = [torch.tensor(ids).to(device=f'cuda:{args.gpu_id}') for ids in stop_words_ids]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    chat = Chat(model, vis_processor, device=f'cuda:{args.gpu_id}', stopping_criteria=stopping_criteria)
    return chat, CONV_VISION



def load_images(image_txt_json: str) -> List[Image.Image]:
    with open(image_txt_json, "r", encoding='utf-8') as f:
        data = json.load(f)
    return data

def create_question_pool() -> List[str]:
    return [
        "Which characteristic morphological features shown in this image support the diagnosis of {}?"
    ]


def process_images(chat: Chat, conv_vision, image_path, question, output_file: str):
    
        
    image = Image.open(image_path)
    chat_state = conv_vision.copy()
    img_list = []
    chat.upload_img(image, chat_state, img_list)
    pred_txt, preds_cls = chat.encode_img(img_list)
    question = question.format(preds_cls[0])
    # print(question)
    chat.ask(question, chat_state, pred_txt[0])
    answer = chat.answer(conv=chat_state, 
                            img_list=img_list,
                            num_beams=1, 
                            temperature=1.0, 
                            max_new_tokens=300, 
                            max_length=2000,
                            min_length= 1)[0]
    answer =  ' '.join([pred_txt[0], answer])
    
    print(f"Image: {image_path}\nQuestion: {question}\nAnswer: {answer}\n"))


def main():
    args = parse_args()
    cfg = Config(args)
    setup_seeds(cfg)
    
    chat, CONV_VISION = initialize_model(args, cfg)
    
    
    questions = create_question_pool()
    output_dir = os.path.dirname(args.output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    process_images(chat, CONV_VISION, args.image_path, questions[0])
    print(f"Processing complete.")

if __name__ == "__main__":
    main()

