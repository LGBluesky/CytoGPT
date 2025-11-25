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
from tqdm import tqdm
import tempfile
import re

def parse_args():
    parser = argparse.ArgumentParser(description="MiniGPT-4_Cyto_Add_head Local Script")
    parser.add_argument("--cfg-path",default='eval_configs/Cyto_minigpt4_eval1.yaml', help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=4, help="specify the gpu to load the model.")
    parser.add_argument("--image-root", default='/mnt/disk1/liushijie/Process/Data/Public/LTCCD/data/origin', help="specify the root of all images")

    parser.add_argument("--image-txt-json", default='/mnt/disk1/liushijie/Process/Data/Public/LTCCD/data/origin/LTCCD_multi-class.json', help="path to image file or directory")
    parser.add_argument("--output-file", default="/mnt/disk1/liushijie/Process/Data/Public/LTCCD/data/origin/generation_test", help="path to output file")
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
        "Based on the category of lesion shown in the image above, what key morphological features of cervical precancerous lesions can be observed?"
    ]



def extract_text_in_parentheses(sentence):
    # 使用正则表达式匹配括号中的内容
    match = re.search(r'\((.*?)\)', sentence)
    if match:
        return match.group(1)  # 返回括号中的内容
    return None  # 如果没有找到括号中的内容，则返回 None



def process_images(chat: Chat, conv_vision, imagedict, questions: List[str], output_path: str, image_root):
    match_num  = 0
    unmatch_num = 0
    print(f"Processing sequence length is : {len(imagedict)}")
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_file = os.path.join(output_path, 'LTCCD_test-1047.json')
    record_list = []
    
    pred_category_dict = {'HSIL': 1, 'LSIL': 2, 'ASCUS': 3, 'ASC-H': 4, 'NILM': 0}
    category_label_dict = {'HSIL': 1, 'LSIL': 2, 'ASCUS': 3, 'ASCH': 4, 'Normal': 0}
    
    
    
    for i, record in tqdm(enumerate(imagedict)):
        
        result = dict()
        image_name  = record['image']
        imagePath = os.path.join(image_root, image_name)
        image = Image.open(imagePath)
        # txtGT = record['caption']
        category = record['category']             
        chat_state = conv_vision.copy()
        img_list = []
        chat.upload_img(image, chat_state, img_list)
        pred_txt = chat.encode_img(img_list)

        result['Image'] = image_name
        result["Index"] = f"{i} / {len(imagedict)}"
        result['Category'] = category
        
        
        catergoy_label = category_label_dict[category]           
    
    
        for question in questions:
  
            if 'negative' in pred_txt[0].lower():
                answer = pred_txt[0]
                pred_category = 0 
            else:
                              
                pred_category_text = extract_text_in_parentheses(pred_txt[0])
                pred_category = pred_category_dict[pred_category_text]
                
                chat.ask(question, chat_state, pred_txt[0])
                answer = chat.answer(conv=chat_state, 
                                    img_list=img_list,
                                    num_beams=1, 
                                    temperature=1.0, 
                                    max_new_tokens=300, 
                                    max_length=2000)[0]
                
                
                answer =  ' '.join([pred_txt[0], answer])
            
            if answer == None:
                match = 0
                unmatch_num +=1
                try:
                    result['Match'] = match
                    result['Q'] = question
                    result['A'] = 'None'
                except:
                    print(f"Error: image name {image_name}")
            else:
                match = (pred_category == catergoy_label)*1
                match_num += match
                try:
                    result['Match'] = match
                    result['Q'] = question
                    result['A'] = answer
                except:
                    print(f"Error: image name {image_name}")
            record_list.append(result)
        
    result_summary = dict()
    result_summary['Total records'] = len(imagedict)
    result_summary['Match record'] = match_num
    result_summary['UnMatch record'] = len(imagedict)-match_num
            
    record_list.append(result_summary)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(record_list, f,  ensure_ascii=False, indent=4)
        
    
def main():
    args = parse_args()
    cfg = Config(args)
    setup_seeds(cfg)
    
    chat, CONV_VISION = initialize_model(args, cfg)
    image_dict = load_images(args.image_txt_json)
    questions = create_question_pool()

    process_images(chat, CONV_VISION, image_dict, questions, args.output_file, args.image_root)
    print(f"Processing complete. Results written to {args.output_file}")

if __name__ == "__main__":
    main()

