from goal_diffusion_0513 import GoalGaussianDiffusion, Trainer
from unet import Unet0526 as Unet
from transformers import CLIPTextModel, CLIPTokenizer
from datasets import SequentialNavDataset
from torch.utils.data import Subset


valid_n = 1
sample_per_seq = 8
target_size = (64, 64)
train_set = SequentialNavDataset(
    sample_per_seq=sample_per_seq, 
    path="../datasets/thor_dataset", 
    target_size=target_size,
)
valid_inds = [i for i in range(0, len(train_set), len(train_set)//valid_n)][:valid_n]
valid_set = Subset(train_set, valid_inds)
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
    train_num_steps =80000,
    save_and_sample_every =5000,
    ema_update_every = 10,
    ema_decay = 0.999,
    train_batch_size =32,
    valid_batch_size =32,
    gradient_accumulate_every = 1,
    num_samples=valid_n, 
    results_folder ='../results_thor',
    fp16 =True,
    amp=True,
)

trainer.train()