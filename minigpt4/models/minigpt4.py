import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from minigpt4.common.registry import registry
from minigpt4.models.base_model import disabled_train
from minigpt4.models.minigpt_base import MiniGPTBase
from minigpt4.models.Qformer import BertConfig, BertLMHeadModel


@registry.register_model("minigpt4")
class MiniGPT4(MiniGPTBase):
    """
    MiniGPT-4 model
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna0": "configs/models/minigpt4_vicuna0.yaml",
        "pretrain_llama2": "configs/models/minigpt4_llama2.yaml",
    }

    def __init__(
            self,
            vit_model="eva_clip_g",
            q_former_model="/mnt/media_nvme/Process/CytoGPT/MiniGPT-4/pretrained_checkpoint/blip2_pretrained_flant5xxl.pth",
            img_size=224,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision="fp16",
            freeze_vit=True,
            has_qformer=True,
            freeze_qformer=True,
            num_query_token=32,
            llama_model="",
            prompt_path="",
            prompt_template="",
            max_txt_len=32,
            end_sym='\n',
            low_resource=False,  # use 8 bit and put vit in cpu
            device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
    ):
        super().__init__(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            llama_model=llama_model,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            prompt_template=prompt_template  # just for conversation fine-tuning 
        )

        self.cls_labels = {0: 'negative', 1: 'atypical squamous cells, cannot exclude high-grade squamous intraepithelial lesion (ASC-H)',
              2: 'atypical squamous cells of undetermined significance (ASCUS)', 3: 'high-grade squamous intraepithelial lesion (HSIL)', 4: 'low-grade squamous intraepithelial lesion (LSIL)'}
        negative_text = [
            # 'This image shows cells classified as negative for intraepithelial malignancy (NILM), reflecting no significant abnormalities.',
            # 'All cells in this image are classified as negative for intraepithelial malignancy (NILM), reflecting no significant abnormalities.',
            # 'This image contains cells classified as negative for intraepithelial malignancy (NILM), reflecting no significant abnormalities.',
            # 'The cells in this image are uniformly classified as negative for intraepithelial malignancy (NILM), reflecting no significant abnormalities.',
            'This image displays cells classified as negative for intraepithelial malignancy (NILM), reflecting no significant abnormalities.'
        ]
        
        postive_txt = [
            'This image shows some cells classified as {}.',
            'Some cells in this image are classified as {}.',
            'This image contains cells that are classified as {}.',
            'Some cells in this image demonstrate features of {}.',
            'This image displays cells that are suggestive of {}.'
        ]
        self.cls_text_mapping = {'negative':negative_text, 'positive':postive_txt}
        
        
        self.has_qformer = has_qformer
        if self.has_qformer:
            print('Loading Q-Former from: {}'.format(q_former_model))
            self.Qformer, self.query_tokens = self.init_Qformer(
                num_query_token, self.visual_encoder.num_features, freeze_qformer
            )
            self.load_from_pretrained(url_or_filename=q_former_model)  # load q-former weights here

            img_f_dim = self.Qformer.config.hidden_size
            print('Loading Q-Former Done')
        else:
            img_f_dim = self.visual_encoder.num_features * 4
            print('Do not use Q-Former here.')

        self.llama_proj = nn.Linear(
            img_f_dim, self.llama_model.config.hidden_size
        )

        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<ImageHere>" in raw_prompt]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []     
        
        
    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, freeze):
        encoder_config = BertConfig.from_pretrained("/mnt/disk1/liushijie/Process/CytoGPT/pretrained_checkpoint/bert-base-uncased/config.json")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 2
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

        Qformer.cls = None
        Qformer.bert.embeddings.word_embeddings = None
        Qformer.bert.embeddings.position_embeddings = None
        for layer in Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        if freeze:
            for name, param in Qformer.named_parameters():
                param.requires_grad = False
            Qformer = Qformer.eval()
            Qformer.train = disabled_train
            query_tokens.requires_grad = False
            logging.info("freeze Qformer")

        return Qformer, query_tokens

    def encode_img(self, image):
        device = image.device
        if len(image.shape) > 4:
            image = image.reshape(-1, *image.shape[-3:])

        with self.maybe_autocast():
            # image_embeds = self.ln_vision(self.visual_encoder(image)).to(device) # default
            image  = image.to(device)
            inter_embedding = self.visual_encoder(image)
            cls_logits = self.cls_head(inter_embedding)
            _, preds = torch.max(cls_logits, 1)
            preds_clss = [self.cls_labels[p.item()] for p in preds]
            preds_txt =[]
            for preds_cls in preds_clss:
                if preds_cls=='negative':
                    preds_txt.append(random.choice(self.cls_text_mapping['negative']))
                else:
                    preds_txt.append(random.choice(self.cls_text_mapping['positive']).format(preds_cls))
            
            
            image_embeds = self.ln_vision(inter_embedding) # 
            
            if self.has_qformer:
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

                query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
                inputs_llama = self.llama_proj(query_output.last_hidden_state)
            else:
                image_embeds = image_embeds[:, 1:, :]
                bs, pn, hs = image_embeds.shape
                image_embeds = image_embeds.view(bs, int(pn / 4), int(hs * 4))

                inputs_llama = self.llama_proj(image_embeds)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
        return inputs_llama, atts_llama, preds_txt, preds_clss

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "MAE_vit_b")
        q_former_model = cfg.get("q_former_model", "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llama_model = cfg.get("llama_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        has_qformer = cfg.get("has_qformer", True)
        freeze_qformer = cfg.get("freeze_qformer", True)
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)

        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')
        print("freeze_qformer: ",freeze_qformer)
        print("vit_model:", vit_model)
        print("max_txt_len: ", max_txt_len)
        print("end_sym: ", end_sym)

        model = cls(
            vit_model=vit_model,
            q_former_model=q_former_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            has_qformer=has_qformer,
            freeze_qformer=freeze_qformer,
            num_query_token=num_query_token,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load MiniGPT-4 Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)

        return model
