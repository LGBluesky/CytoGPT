"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from minigpt4.datasets.datasets.caption_datasets import CaptionDataset,  CaptionEvalDataset

class CytoCapDataset(CaptionDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        # img_file = '{:0>12}.jpg'.format(ann["image_id"]) # origin minigpt4
        # img_file = ann["image_id"]
        img_file = ann["image"]
        image_path = os.path.join(self.vis_root, img_file)
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        # pre_caption = 'This image shows cells with {}. '.format(ann['category'])
        # caption = self.text_processor(pre_caption+ann["caption"])
        caption = self.text_processor(ann["caption"])

        return {
            "image": image,
            "answer": caption,
            "image_id": self.img_ids[ann["image_id"]],
        }



class CytoCapEvalDataset(CaptionEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        img_id = ann["image_id"]

        return {
            "image": image,
            "image_id": img_id,
            "instance_id": '0',  ## Cyto Dataset have no instance_id.
        }


class CytoNoCapsEvalDataset(CaptionEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]
        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        img_id = ann["img_id"]

        return {
            "image": image,
            "image_id": img_id,
            "instance_id": '0',   # Cyto Dataset have no instance_id.
        }
