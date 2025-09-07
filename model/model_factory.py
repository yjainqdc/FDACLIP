from config.base_config import Config
from model.clip_baseline import CLIPBaseline
from model.clip_transformer import CLIPTransformer
from model.prompt_clip import PromptCLIP
from model.FrameCLIP import FrameCLIP
from model.AlphaCLIP_Video import AlphaCLIP_Video
from model.AlphaCLIP_Video_transPOOL import AlphaCLIP_Video_transformer

class ModelFactory:
    @staticmethod
    def get_model(config: Config):
        print(type(config.arch))
        print(config.arch)
        if config.arch == 'clip_baseline':
            return CLIPBaseline(config)
        elif config.arch == 'clip_transformer':
            return CLIPTransformer(config)
        elif config.arch == 'prompt_clip':
            return PromptCLIP(config)
        elif config.arch == 'prompt_clip_baseline':
            return PromptCLIP(config)
        elif config.arch == 'frame_clip':
            return FrameCLIP(config)
        elif config.arch == 'alphaclip_video':
            return AlphaCLIP_Video(config)
        elif config.arch == 'alphaclip_video_transformer':
            return AlphaCLIP_Video_transformer(config)
        else:
            raise NotImplemented
