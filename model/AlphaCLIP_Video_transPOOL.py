from model import clip_model
from torch import nn
#from config.base_config import Config
import torch
from modules.baseline_pooling import BaselinePooling
from modules.transformer import Transformer
from model import alpha_clip


device = "cuda" if torch.cuda.is_available() else "cpu"

class AlphaCLIP_Video_transformer(nn.Module):
    '''
    embeddings应该在创建模型时传入类别向量（GCN前）
    然后每次forword会再次传入新的（GCN后）
    后面还会将GCN挂到模型实体熵
    '''
    def __init__(self, config):
        super(AlphaCLIP_Video_transformer, self).__init__()
        self.config = config
        #self.clip = CLIPModel.from_pretrained("/sshfs/jiaao/workdir/CLIP_test/PromptSwitch-main/CLIP_Vit_32/")
        self.clip, self.preprocess = alpha_clip.load("ViT-B/16", device=device,lora_adapt=True, rank=16)  # change to your own ckpt path
        self.clip = self.clip.float()
        # assert config.pooling_type == config.pooling_type_test
        # self.pool_frames = BaselinePooling(config.pooling_type, config)
        # self.pool_frames_test = self.pool_frames
        config.pooling_type = 'transformer'
        config.pooling_type_test = 'transformer'
        #assert config.pooling_type == config.pooling_type_test
        self.pool_frames = Transformer(config)
        self.pool_frames_test = self.pool_frames

        params_optimizer = list(self.named_parameters())
        self.clip_params = [p for n, p in params_optimizer if "clip." in n]
        self.noclip_params = [p for n, p in params_optimizer if "clip." not in n]

        # for param in self.clip.transformer.parameters():
        #     param.requires_grad = False


    def forward(self, data, return_all_frames=False):
        batch_size = data['video'].shape[0]
        text_data = data['text']
        video_data = data['video']
        mask_data = data['diff_mask']

        # print(text_data)

        video_data = video_data.reshape(-1, 3, self.config.input_res, self.config.input_res).half()
        mask_data = mask_data.reshape(-1, 1, self.config.input_res, self.config.input_res).half()
        text_features = self.clip.encode_text(text_data)
        video_features = self.clip.visual(video_data.float(), mask_data.float())


        #video_data = video_data.reshape(-1, 3, self.config.input_res, self.config.input_res)
        # text_features = self.clip.get_text_features(**text_data)
        # video_features = self.clip.get_image_features(video_data)
        # video_features = video_features.reshape(batch_size, self.config.num_frames, -1)

        # YJA+临时
        video_features = video_features.reshape(batch_size, -1, video_features.size(-1))
        video_features_pooled = self.pool_frames(text_features.float(), video_features.float())

        # if return_all_frames:
        #     return text_features, video_features, video_features_pooled

        # return text_features, video_features_pooled
        output = {'text_features': text_features.float()}
        if return_all_frames:
            output['video_features'] = video_features.float()
        output['video_features_pooled'] = video_features_pooled.float()

        return output



# CLIP, preprocess = clip_model.load("ViT-B/32", device=device)
# video = torch.randn(64,6,3,224,224).half().cuda()
# x,_ = CLIP.encode_image_video(video)
# print(x.shape)