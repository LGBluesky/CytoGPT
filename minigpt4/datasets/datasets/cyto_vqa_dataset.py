"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json
import random

from PIL import Image

from minigpt4.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset

from collections import OrderedDict
from torch.utils.data import Dataset



class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "question": ann["question"],
                "question_id": ann["question_id"],
                "answers": "; ".join(ann["answer"]),
                "image": sample["image"],
            }
        )


class CytoVQADataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        # default settings
        # self.system_instruction_pool =[
        #     "[vqa] {}",
        #     "[vqa] Based on the image, respond to this question with a short answer: {}"
        # ]

        self.prompt_template='###Human: {} ###Assistant: '

        # self.system_instruction_pool = "Based on the image, respond to this question with a short answer: {}"  # first setting
        # self.system_instruction_pool = "{}"
        # self.custom_instruction_for_feature=[
        #     "Based on the category of lesion shown in the image above, what key morphological features of cervical precancerous lesions can be observed?",
        #     "Based on the category of lesion shown in the image above, what key morphological features of cervical precancerous lesions are visible?",
        #     "Based on the category of lesion shown in the image above, what key morphological  characteristics of cervical precancerous lesions can be observed?",
        #     "Based on the category of lesion shown in the image above, What key morphological characteristics of cervical precancerous lesions are visible?"
        # ]
        
        # self.system_instruction_pool = "You are a cytopathologist. Please analyze cervical cytology images according to the 2014 Bethesda System (TBS). {}"
        self.system_instruction_pool = "{}"
        self.custom_instruction_for_feature =  "{} Which characteristic morphological features shown in this image support the diagnosis of {}?"
        


        exist_annotation = []
        for ann in self.annotation:
            image_path = os.path.join(self.vis_root, ann["image"])
            if os.path.exists(image_path):
                exist_annotation.append(ann)
        self.annotation = exist_annotation


    def get_data(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        answer = ann["caption"]

        return {
            "image": image,
            "answer": answer,
            "category": ann["category"]
        }

    def __getitem__(self, index):
        data = self.get_data(index)
        thre = 0.6
        # if random.random() < thre:
        #     custom_instruction = random.choice(self.custom_instruction_pool_W_).format(data['category']) # new added
        # else:
        #     custom_instruction = 'This is a cervical cytological image of {}. '.format(data['category'])+ random.choice(self.custom_instruction_pool_WO_)

        
        ## default setting
        # custom_instruction = f"The image shows a cervical cell with features consistent with {data['category']}"+ random.choice(self.custom_instruction_pool_WO_)
        # instruction = self.system_instruction_pool.format(custom_instruction)
        # instruction = "<Img><ImageHere></Img>  ".format(instruction)
        # instruction = self.prompt_template.format(instruction)

        
        ## for new category prompt
        instruction = self.system_instruction_pool.format(self.custom_instruction_for_feature)

        instruction = "<Img><ImageHere></Img>" + instruction

        return {
            "image": data['image'],
            "question_id": 0,  # without question_id
            "instruction_input": instruction,
            "answer": self.text_processor(data['answer'])
        }


class CytoMultiVQADataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        # super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.vis_root = vis_root

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.prompt_template='###Human: {} ###Assistant: '
        # self.system_instruction_pool = "Based on the image, respond to this question with a short answer: {}"  # first setting
        self.system_instruction_pool = "{}"

        # exist_annotation = []
        # for ann in self.annotation:
        #     image_path = os.path.join(self.vis_root, ann["image"])
        #     if os.path.exists(image_path):
        #         exist_annotation.append(ann)
        # self.annotation = exist_annotation

        print(f"loading annotations from {ann_paths}")
        with open(ann_paths, 'r', encoding= 'utf-8') as f:
            self.annotation = json.load(f)

    def __len__(self):
        return len(self.annotation)


    def get_data(self, index):
        ann = self.annotation[index]
        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        answer = ann["caption"]
        questions = ann["question"]

        return {
            "image": image,
            "question":questions,
            "answer": answer,
            "category": ann["category"]
        }

    def __getitem__(self, index):
        data = self.get_data(index)
    
        instruction = self.system_instruction_pool.format(data["question"])
        instruction = "<Img><ImageHere></Img> {} ".format(instruction)
        instruction = self.prompt_template.format(instruction)

        return {
            "image": data['image'],
            "question_id": 0,  # without question_id
            "instruction_input": instruction,
            "answer": self.text_processor(data['answer'])
        }


## just for debug
if __name__ == '__main__':
    vis_root = '/mnt/media_nvme/Process/CytoGPT/Data/Cyto/sfy1_part'
    ann_paths = ["/mnt/media_nvme/Process/CytoGPT/Data/Cyto/sfy1_part/all_labels_W_class/train.json"]
    cyto_vqa = CytoVQADataset(None, None, vis_root, ann_paths)
    for i in range(5):
        combine_dict = cyto_vqa.__getitem__(i)
        print(combine_dict['instruction_input'])

