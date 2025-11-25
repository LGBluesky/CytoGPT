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
    parser.add_argument("--gpu-id", type=int, default=5, help="specify the gpu to load the model.")
    parser.add_argument("--image-root", default='/mnt/disk1/liushijie/Process/Data/Q-former_finetune_data/Auto_revise/NewData/image', help="specify the root of all images")

    parser.add_argument("--image-txt-json", default='/mnt/disk1/liushijie/Process/Data/Q-former_finetune_data/Auto_revise/NewData/ManualRevise/test_all.json', help="path to image file or directory")
    parser.add_argument("--output-file", default="/mnt/disk1/liushijie/Process/CytoGPT/MiniGPT-4_Cyto-result/minigpt4_autoPrompt_finetune/20250313155/generation_test/test_all.txt", help="path to output file")
    parser.add_argument("--temperature", default=1.0, type=float, help="temperature for generation")
    parser.add_argument("--num_beams", default=1, type=int, help="number of beams for beam search")

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
    # return [
    #     "Based on the category of lesion shown in the image above, what key morphological features of cervical precancerous lesions can be observed?"
    # ] #test1
    # return [
    #     "What are the key morphological features of the following cervical cell image according to the Bethesda System (TBS) criteria?"
    # ] #test2
    
    
    return [
        "Which characteristic morphological features are shown in this image support the diagnosis of {}?"
    ]


def abstract_keyword_in_a(a):
    a = a.lower()
    
    # 使用正则表达式进行更精确的匹配
    # 检查括号中的缩写
    nilm_pattern = re.search(r'\(nilm\)', a)
    hsil_pattern = re.search(r'\(hsil\)', a)
    lsil_pattern = re.search(r'\(lsil\)', a)
    ascus_pattern = re.search(r'\(asc-?us\)', a)  # 匹配(asc-us)或(ascus)
    asch_pattern = re.search(r'\(asch\)', a)
    
    if nilm_pattern:
        return 'NILM'
    if hsil_pattern:
        return 'HSIL'
    if lsil_pattern:
        return 'LSIL'
    if ascus_pattern:
        return 'ASCUS'
    if asch_pattern:
        return 'ASCH'
    
    # 检查完整短语或独立术语
    if re.search(r'\bhsil\b', a) or re.search(r'high-grade squamous intraepithelial lesion', a):
        return 'HSIL'
    elif re.search(r'\blsil\b', a) or re.search(r'low-grade squamous intraepithelial lesion', a):
        return 'LSIL'
    elif re.search(r'\basc-?us\b', a) or re.search(r'atypical squamous cells of undetermined significance', a):
        return 'ASCUS'
    elif re.search(r'\basch\b', a) or re.search(r'atypical squamous cells,? cannot exclude hsil', a):
        return 'ASCH'
    elif re.search(r'negative for intraepithelial', a) or re.search(r'\bnilm\b', a):
        return 'NILM'
    else:
        return 'Unknown'
def extract_category_keyword(sentence):
    """
    从句子中提取'as'或'of'后面的类别关键词
    
    参数:
        sentence (str): 输入的句子
    
    返回:
        str or None: 提取到的类别关键词，如果没有找到则返回None
    
    要求:
        1. 句子中作为连接词的'as'或'of'只能出现一次
        2. 提取'as'或'of'后面直到'.'之前的内容作为类别
        3. 类别名称本身可能包含'of'（如ASCUS）
        4. 使用正则表达式实现
    """
    
    # 使用更精确的正则表达式来匹配连接词
    # 查找 "classified as" 或 "suggestive of" 等模式
    # 模式说明:
    # (?:classified\s+as|suggestive\s+of)\s+ - 匹配"classified as"或"suggestive of"后跟空格
    # ([^.]+) - 捕获后面直到'.'之前的所有字符作为类别
    # \. - 匹配句号
    pattern = r'(?:classified\s+as|suggestive\s+of)\s+([^.]+)\.'
    
    match = re.search(pattern, sentence, re.IGNORECASE)
    
    if match:
        # 提取并清理关键词（去除首尾空格）
        keyword = match.group(1).strip()
        return keyword
    
    # 如果上面的模式没匹配到，尝试更通用的模式
    # 匹配单词边界后的 as 或 of（避免匹配类别名称中的of）
    general_pattern = r'\b(?:as|of)\s+([^.]+)\.'
    
    match = re.search(general_pattern, sentence, re.IGNORECASE)
    
    if match:
        keyword = match.group(1).strip()
        # 检查这个匹配是否合理（不是类别内部的of）
        # 通过检查前面的上下文来判断
        before_match = sentence[:match.start()].lower()
        if any(word in before_match for word in ['classified', 'suggestive', 'features']):
            return keyword
    
    print("未找到匹配的模式")
    return None
    
    
    
def check_keyword(a, category):
    """检查A中是否包含指定关键字"""
    a_category = abstract_keyword_in_a(a)    
    return 1 if category.lower() == a_category.lower() else 0


def process_images(chat: Chat, conv_vision, imagedict, questions: List[str], output_file: str, image_root, args):
    match_num  = 0
    unmatch_num = 0
    print(f"Processing sequence length is : {len(imagedict)}")
    with open(output_file, 'w') as f:

        for i, record in tqdm(enumerate(imagedict)):
            image_name  = record['image']
            imagePath = os.path.join(image_root, image_name)
            image = Image.open(imagePath)
            txtGT = record['caption']
            category = record['category']
            record_index = record["record_index"]
            ori_score = record["accuracy_score"]
                         
            chat_state = conv_vision.copy()
            img_list = []
            chat.upload_img(image, chat_state, img_list)
            pred_txt, preds_cls = chat.encode_img(img_list)


            f.write(f"Image: {image_name}\n")
            f.write(f"record_index: {record_index} \n")
            f.write(f"Category: {category}\n")
            f.write(f"Ori_geval_score: {ori_score}\n")
            
            
            for question in questions:
                # if category != 'Unknown':
                #     # question  = f'This is a cervical cytological image of {category}. ' + question
                #     question  = f'The image shows a cervical cell with features consistent with {category}. ' + question

                #     pass
                # else:
                #     print(image_name)
                question = question.format(preds_cls[0])
                # print(question)
                chat.ask(question, chat_state, pred_txt[0])
                answer = chat.answer(conv=chat_state, 
                                     img_list=img_list,
                                     num_beams=args.num_beams, 
                                     temperature=args.temperature, 
                                     max_new_tokens=300, 
                                     max_length=2000,
                                     min_length= 1)[0]
                answer =  ' '.join([pred_txt[0], answer])
                if answer == None:
                    match = 0
                    unmatch_num +=1
                    try:
                        f.write(f"Match: {match}\n\n")
                        f.write(f"GT: {txtGT}\n")
                        f.write(f"Q: {question}\n")
                        f.write(f"A: {answer}\n\n")
                    except:
                        print(f"Error: image name {image_name}")

                else:
                    match = check_keyword(answer, category)
                    match_num += match
                    try:
                        f.write(f"Match: {match}\n\n")
                        f.write(f"GT: {txtGT}\n")
                        f.write(f"Q: {question}\n")
                        f.write(f"A: {answer}\n\n")
                    except:
                        print(f"Error: image name {image_name}")

            f.write("\n" + "="*50 + "\n\n")

        summary = f"Total records: {len(imagedict)}\n"
        summary += f"Match record: {match_num}\n"
        summary += f"UnMatch record: {len(imagedict)-match_num}\n"
        print(summary)

    
            
        f.write("="*25 +' Summary  '+ "="*25+"\n\n")
        f.write(summary)


def main():
    args = parse_args()
    cfg = Config(args)
    setup_seeds(cfg)
    
    chat, CONV_VISION = initialize_model(args, cfg)
    image_dict = load_images(args.image_txt_json)
    questions = create_question_pool()
    output_dir = os.path.dirname(args.output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    process_images(chat, CONV_VISION, image_dict, questions, args.output_file, args.image_root, args)
    print(f"Processing complete. Results written to {args.output_file}")

if __name__ == "__main__":
    main()

