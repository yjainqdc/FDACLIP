import sys
sys.path.append('./')

import os
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter
from transformers.optimization import get_cosine_schedule_with_warmup
from torch.optim import AdamW
from config.all_config import AllConfig
from video_datasets.data_factory import DataFactory
from model.model_factory import ModelFactory
from modules.loss import LossFactory
from trainer.trainer import Trainer

import torch.distributed as dist
from torch.nn.parallel import DataParallel



def main():
    config = AllConfig()

    assert config.num_frames % config.num_prompts == 0
    assert config.num_test_frames % config.num_prompts == 0

    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    if not config.no_tensorboard:
        writer = SummaryWriter(log_dir=config.tb_log_dir)
    else:
        writer = None

    if config.seed >= 0:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        random.seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model = ModelFactory.get_model(config)
    #多卡训练
    #model = DataParallel(model, device_ids=[0,1,2,3])



    train_data_loader = DataFactory.get_data_loader(config, split_type='train')
    valid_data_loader  = DataFactory.get_data_loader(config, split_type='test')
    # train_data_loader = DataFactory.get_data_loader_DDP(config, split_type='train')
    # valid_data_loader  = DataFactory.get_data_loader_DDP(config, split_type='test')



    optimizer_grouped_params = [
        {'params': model.clip_params, 'lr': config.clip_lr},
        {'params': model.noclip_params, 'lr': config.noclip_lr}
    ]
    optimizer = AdamW(optimizer_grouped_params, weight_decay=config.weight_decay)
    num_training_steps = len(train_data_loader) * config.num_epochs
    num_warmup_steps = int(config.warmup_proportion * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)

    loss = LossFactory.get_loss(config)

    trainer = Trainer(model, loss, optimizer,
                      config=config,
                      train_data_loader=train_data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=scheduler,
                      writer=writer,
                      use_ema=config.use_ema)

    trainer.train()


if __name__ == '__main__':
    main()
