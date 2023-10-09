from goal_diffusion_0411 import GoalGaussianDiffusion, Trainer
from unet import Unet0411 as Unet
from transformers import CLIPTextModel, CLIPTokenizer
import torch
import numpy as np
import os 
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
import json
from PIL import Image
import tqdm
from accelerate import Accelerator
from datasets import SequentialDatasetNp, SequentialDatasetVal


sample_per_seq = 7
target_size = (48, 64)
train_set = SequentialDatasetNp(
    sample_per_seq=sample_per_seq, 
    path="../datasets/numpy/bridge_data_v1/berkeley/", 
    target_size=target_size,
    debug=False,
)
valid_set = SequentialDatasetVal(
    sample_per_seq=sample_per_seq, 
    path="../datasets/valid",
    target_size=target_size,
)
Unet = Unet()

pretrained_model = "openai/clip-vit-base-patch32"
tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
text_encoder = CLIPTextModel.from_pretrained(pretrained_model)
text_encoder.requires_grad_(False)
text_encoder.eval()

diffusion = GoalGaussianDiffusion(
    channels=3*(sample_per_seq-1),
    model=Unet,
    image_size=target_size,
    timesteps=100,
    sampling_timesteps=100,
    loss_type='l2',
    objective='pred_noise',
    beta_schedule = 'cosine',
    min_snr_loss_weight = False,
)

trainer = Trainer(
    diffusion_model=diffusion,
    tokenizer=tokenizer, 
    text_encoder=text_encoder,
    train_set=train_set,
    valid_set=valid_set,
    train_lr=1e-4,
    train_num_steps =180000,
    save_and_sample_every =4000,
    ema_update_every = 10,
    ema_decay = 0.999,
    train_batch_size =32,
    valid_batch_size =30,
    gradient_accumulate_every = 1,
    num_samples=30, 
    results_folder ='../results_bridge',
    fp16 =True,
    amp=True,
)

trainer.train()

