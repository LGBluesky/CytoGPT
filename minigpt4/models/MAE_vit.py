from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer
import copy

def convert_weights_to_fp16(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

#         if isinstance(l, (nn.MultiheadAttention, Attention)):
#             for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
#                 tensor = getattr(l, attr)
#                 if tensor is not None:
#                     tensor.data = tensor.data.half()

    model.apply(_convert_weights_to_fp16)

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)
            del self.norm  # remove the original norm
        self.head = nn.Identity()
        self.num_features = self.embed_dim    # new add

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        return x

def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


class linear_classifier(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super(linear_classifier, self).__init__()
        self.BN = nn.BatchNorm1d(embed_dim)
        self.fc1 = nn.Linear(embed_dim,embed_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(embed_dim,num_classes)
    def forward(self, x):
        x = self.BN(x)
        x = self.fc1(x)        
        x = self.activation(x)        
        x = self.dropout(x)        
        x = self.fc2(x)        
        return x        
    

class MAE_transform_cls_part(timm.models.vision_transformer.VisionTransformer):
    def __init__(self,out_dim, transformer_layer=6, **kwargs):
        super(MAE_transform_cls_part, self).__init__(**kwargs)
        self.transformer_layer=transformer_layer
        # self.model2 = nn.Sequential(*list(self.blocks.children())[:self.transformer_layer])
        self.model2 = nn.Sequential(*[copy.deepcopy(block) for block in list(self.blocks.children())[:self.transformer_layer]])
        
        self.class_head = linear_classifier(768, out_dim)
        
    def forward_features(self, x):
        x = self.model2(x)
        x = self.norm(x)
        outcome = x[:, 0]        #  cls token as global feature
        return outcome
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.class_head(x)
        return  x




def create_mae_vit_b(img_size=224,use_checkpoint=False, precision="fp16"):
    model = vit_base_patch16()
    url = '/mnt/disk1/liushijie/Process/CytoGPT/pretrained_checkpoint/MAE_vit_Checkpoint/checkpoint-19_model_state.pth'
    print('load image encoder from:' +url)
    state_dict = torch.load(url, map_location="cpu")
    interpolate_pos_embed(model, state_dict)
    incompatible_keys = model.load_state_dict(state_dict, strict=False)
    # print(incompatible_keys)
    # output2 = model(img)
    # print(output2.size())
    if precision == "fp16":
        convert_weights_to_fp16(model)
    return model

def create_cls_head(num_class=5):
    model = MAE_transform_cls_part(out_dim=num_class, transformer_layer=6)
    url =  '/mnt/disk1/liushijie/Process/CytoGPT/pretrained_checkpoint/cls_head/checkpoint_499_slim1.pth'# 记得转换
    state_dict = torch.load(url, map_location="cpu")["model"]
    msg = model.load_state_dict(state_dict)
    print(f"Class head Missing keys {msg}")
    
    return model

if __name__ == '__main__':
    # import torch
    # img = torch.randn(2, 3, 224, 224)
    # # img = img.type(torch.half)
    # img = img.cuda()
    # model = create_mae_vit_b()
    # model = model.cuda()
    # with torch.autocast(device_type='cuda'):
    #     output3 = model(img)
    # print(output3.size())
    # print('OK')
    src = r""