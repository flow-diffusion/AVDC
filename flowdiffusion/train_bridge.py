from goal_diffusion import GoalGaussianDiffusion, Trainer
from unet import UnetBridge as Unet
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
import argparse


def main(args):
    valid_n = 1
    sample_per_seq = 7
    target_size = (48, 64)

    if args.mode == 'inference':
        train_set = valid_set = [None] # dummy
    else:
        train_set = SequentialDatasetNp(
            sample_per_seq=sample_per_seq, 
            path="../datasets/bridge/numpy/bridge_data_v1/berkeley/", 
            target_size=target_size,
            debug=False,
        )
        valid_inds = [i for i in range(0, len(train_set), len(train_set)//valid_n)][:valid_n]
        valid_set = Subset(train_set, valid_inds)
    unet = Unet()

    pretrained_model = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model)
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    diffusion = GoalGaussianDiffusion(
        channels=3*(sample_per_seq-1),
        model=unet,
        image_size=target_size,
        timesteps=100,
        sampling_timesteps=args.sample_steps,
        loss_type='l2',
        objective='pred_v',
        beta_schedule = 'cosine',
        min_snr_loss_weight = True,
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
        valid_batch_size =valid_n,
        gradient_accumulate_every = 1,
        num_samples=30, 
        results_folder ='../results/bridge',
        fp16 =True,
        amp=True,
    )

    if args.checkpoint_num is not None:
        trainer.load(args.checkpoint_num)
    
    if args.mode == 'train':
        trainer.train()
    else:
        from PIL import Image
        from torchvision import transforms
        import imageio
        import torch
        from os.path import splitext
        text = args.text
        image = Image.open(args.inference_path)
        batch_size = 1
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ])
        image = transform(image)
        output = trainer.sample(image.unsqueeze(0), [text], batch_size).cpu()
        output = output[0].reshape(-1, 3, *target_size)
        output = torch.cat([image.unsqueeze(0), output], dim=0)
        root, ext = splitext(args.inference_path)
        output_gif = root + '_out.gif'
        output = (output.cpu().numpy().transpose(0, 2, 3, 1).clip(0, 1) * 255).astype('uint8')

        ## 231130 resize output image to 240x320 to make it look better
        output = [np.array(Image.fromarray(frame).resize((320, 240))) for frame in output]

        imageio.mimsave(output_gif, output, duration=200, loop=1000)
        print(f'Generated {output_gif}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default='train', choices=['train', 'inference']) # 'train for training, 'inference' for generating samples
    parser.add_argument('-c', '--checkpoint_num', type=int, default=None) # checkpoint number to resume training or generate samples
    parser.add_argument('-p', '--inference_path', type=str, default=None) # path to input image
    parser.add_argument('-t', '--text', type=str, default=None) # task text 
    parser.add_argument('-g', '--guidance_weight', type=int, default=0) # set to positive to use guidance
    args = parser.parse_args()
    if args.mode == 'inference':
        assert args.checkpoint_num is not None
        assert args.inference_path is not None
        assert args.text is not None
        assert args.sample_steps <= 100
    main(args)

