from model import clip_model
from torch import nn
#from config.base_config import Config
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"

class FrameCLIP(nn.Module):
    '''
    embeddings应该在创建模型时传入类别向量（GCN前）
    然后每次forword会再次传入新的（GCN后）
    后面还会将GCN挂到模型实体熵
    '''
    def __init__(self, config):
        super(FrameCLIP, self).__init__()
        self.config = config
        self.clip, preprocess = clip_model.load("ViT-B/32", device=device)
        params_optimizer = list(self.named_parameters())
        self.clip_params = [p for n, p in params_optimizer if "VideoCatFramePooling" not in n]
        self.noclip_params = [p for n, p in params_optimizer if "VideoCatFramePooling" in n]
        config.pooling_type = 'avg'

    def forward(self, data, return_all_frames=False):
        #batch_size = data['video'].shape[0]
        text_data = data['text']
        video_data = data['video']

        text_features = self.clip.encode_text(text_data)
        video_features,video_features_all = self.clip.encode_image_video(video_data)

        output = {'text_features': text_features}
        if return_all_frames:
            output['video_features'] = video_features_all
        output['video_features_pooled'] = video_features

        return output



# CLIP, preprocess = clip_model.load("ViT-B/32", device=device)
# video = torch.randn(64,6,3,224,224).half().cuda()
# x,_ = CLIP.encode_image_video(video)
# print(x.shape)