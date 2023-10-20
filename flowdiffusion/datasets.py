from torch.utils.data import Dataset
import os
from glob import glob
import torch
from utils import get_paths, get_paths_from_dir
from tqdm import tqdm
from PIL import Image
import numpy as np
import json
import torchvision.transforms as T
import random
from torchvideotransforms import video_transforms, volume_transforms
from einops import rearrange
# from vidaug import augmentors as va

random.seed(0)

### Sequential Datasets: given first frame, predict all the future frames

class SequentialDatasetNp(Dataset):
    def __init__(self, path="../datasets/numpy/bridge_data_v1/berkeley", sample_per_seq=7, debug=False, target_size=(128, 128)):
        print("Preparing dataset...")
        self.sample_per_seq = sample_per_seq

        sequence_dirs = glob(os.path.join(path, "**/out.npy"), recursive=True)
        if debug:
            sequence_dirs = sequence_dirs[:10]
        self.sequences = []
        self.tasks = []
    
        obss, tasks = [], []
        for seq_dir in tqdm(sequence_dirs):
            obs, task = self.extract_seq(seq_dir)
            tasks.extend(task)
            obss.extend(obs)

        self.sequences = obss
        self.tasks = tasks
        self.transform = T.Compose([
            T.Resize(target_size),
            T.ToTensor()
        ])
        print("training_samples: ", len(self.sequences))
        print("Done")

    def extract_seq(self, seqs_path):
        seqs = np.load(seqs_path, allow_pickle=True)
        task = seqs_path.split('/')[-3].replace('_', ' ')
        outputs = []
        for seq in seqs:
            observations = seq["observations"]
            viewpoints = [v for v in observations[0].keys() if "image" in v]
            N = len(observations)
            for viewpoint in viewpoints:
                full_obs = [observations[i][viewpoint] for i in range(N)]
                sampled_obs = self.get_samples(full_obs)
                outputs.append(sampled_obs)
        return outputs, [task] * len(outputs)

    def get_samples(self, seq):
        N = len(seq)
        ### uniformly sample {self.sample_per_seq} frames, including the first and last frame
        samples = []
        for i in range(self.sample_per_seq-1):
            samples.append(int(i*(N-1)/(self.sample_per_seq-1)))
        samples.append(N-1)
        return [seq[i] for i in samples]
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        samples = self.sequences[idx]
        # images = [torch.FloatTensor(np.array(Image.open(s))[::4, ::4].transpose(2, 0, 1) / 255.0) for s in samples]
        images = [self.transform(Image.fromarray(s)) for s in samples]
        x_cond = images[0] # first frame
        x = torch.cat(images[1:], dim=0) # all other frames
        task = self.tasks[idx]
        return x, x_cond, task
        
class SequentialDataset(SequentialDatasetNp):
    def __init__(self, path="../datasets/frederik/berkeley", sample_per_seq=7, target_size=(128, 128)):
        print("Preparing dataset...")
        sequence_dirs = get_paths(path)
        self.sequences = []
        self.tasks = []
        for seq_dir in tqdm(sequence_dirs):
            seq = self.get_samples(get_paths_from_dir(seq_dir))
            if len(seq) > 1:
                self.sequences.append(seq)
            task = seq_dir.split('/')[-6].replace('_', ' ')
            self.tasks.append(task)
        self.sample_per_seq = sample_per_seq
        self.transform = T.Compose([
            T.Resize(target_size),
            T.ToTensor()
        ])
        print("Done")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        samples = self.sequences[idx]
        images = [self.transform(Image.open(s)) for s in samples]
        x_cond = images[0] # first frame
        x = torch.cat(images[1:], dim=0) # all other frames
        task = self.tasks[idx]
        return x, x_cond, task

class SequentialDatasetVal(SequentialDataset):
    def __init__(self, path="../datasets/valid", sample_per_seq=7, target_size=(128, 128)):
        print("Preparing dataset...")
        sequence_dirs = sorted([d for d in os.listdir(path) if "json" not in d], key=lambda x: int(x))
        self.sample_per_seq = sample_per_seq
        self.sequences = []
        self.tasks = []
        for seq_dir in tqdm(sequence_dirs):
            seq = self.get_samples(get_paths_from_dir(os.path.join(path, seq_dir)))
            if len(seq) > 1:
                self.sequences.append(seq)
            
        with open(os.path.join(path, "valid_tasks.json"), "r") as f:
            self.tasks = json.load(f)
        self.transform = T.Compose([
            T.Resize(target_size),
            T.ToTensor()
        ])
        print("Done")

### Markovian datasets: given current frame, predict the next frame
class MarkovianDatasetNp(SequentialDatasetNp):
    def __getitem__(self, idx):
        samples = self.sequences[idx]
        ### random sample 2 consecutive frames
        start_ind = np.random.randint(0, len(samples)-1)
        x_cond = torch.FloatTensor(samples[start_ind].transpose(2, 0, 1) / 255.0)
        x = torch.FloatTensor(samples[start_ind+1].transpose(2, 0, 1) / 255.0)
        task = self.tasks[idx]
        return x, x_cond, task
    
    def get_first_frame(self, idx):
        samples = self.sequences[idx]
        return torch.FloatTensor(samples[0].transpose(2, 0, 1) / 255.0)
    
class MarkovianDatasetVal(SequentialDatasetVal):
    def __getitem__(self, idx):
        samples = self.sequences[idx]
        ### random sample 2 consecutive frames
        start_ind = np.random.randint(0, len(samples)-1)
        x_cond = self.transform(Image.open(samples[start_ind]))
        x = self.transform(Image.open(samples[start_ind+1]))
        task = self.tasks[idx]
        return x, x_cond, task
    
    def get_first_frame(self, idx):
        samples = self.sequences[idx]
        return torch.FloatTensor(Image.open(samples[0]))
        
class AutoregDatasetNp(SequentialDatasetNp):
    def __getitem__(self, idx):
        samples = self.sequences[idx]
        pred_idx = np.random.randint(1, len(samples))
        images = [torch.FloatTensor(s.transpose(2, 0, 1) / 255.0) for s in samples]
        x_cond = torch.cat(images[:-1], dim=0)
        x_cond[:, 3*pred_idx:] = 0.0
        x = images[pred_idx]
        task = self.tasks[idx]
        return x, x_cond, task
        
class AutoregDatasetNpL(SequentialDatasetNp):
    def __getitem__(self, idx):
        samples = self.sequences[idx]
        N = len(samples)
        h, w, c = samples[0].shape
        pred_idx = np.random.randint(1, N)
        images = [torch.FloatTensor(s.transpose(2, 0, 1) / 255.0) for s in samples]
        x_cond = torch.zeros((N-1)*c, h, w)
        x_cond[(N-pred_idx-1)*3:] = torch.cat(images[:pred_idx])
        x = images[pred_idx]
        task = self.tasks[idx]
        return x, x_cond, task
    
# SSR datasets
class SSRDatasetNp(SequentialDatasetNp):
    def __init__(self, path="../datasets/numpy/bridge_data_v1/berkeley", sample_per_seq=7, debug=False, target_size=(128, 128), in_size=(48, 64), cond_noise=0.2):
        super().__init__(path, sample_per_seq, debug, target_size)
        self.downsample_tfm = T.Compose([
            T.Resize(in_size),
            T.Resize(target_size),
            T.ToTensor()
        ])

    def __getitem__(self, idx):
        samples = self.sequences[idx]
        # images = [torch.FloatTensor(np.array(Image.open(s))[::4, ::4].transpose(2, 0, 1) / 255.0) for s in samples]
        x = torch.cat([self.transform(Image.fromarray(s)) for s in samples][1:], dim=0)
        x_cond = torch.cat([self.downsample_tfm(Image.fromarray(s)) for s in samples][1:], dim=0)
        ### apply noise on x_cond
        cond_noise = torch.randn_like(x_cond) * 0.2
        x_cond = x_cond + cond_noise
        task = self.tasks[idx]
        return x, x_cond, task
    
class SSRDatasetVal(SequentialDatasetVal):
    def __init__(self, path="../datasets/valid", sample_per_seq=7, target_size=(128, 128), in_size=(48, 64)):
        print("Preparing dataset...")
        super().__init__(path, sample_per_seq, target_size)
        self.downsample_tfm = T.Compose([
            T.Resize(in_size),
            T.Resize(target_size),
            T.ToTensor()
        ])
    def __getitem__(self, idx):
        samples = self.sequences[idx]
        # images = [torch.FloatTensor(np.array(Image.open(s))[::4, ::4].transpose(2, 0, 1) / 255.0) for s in samples]
        x = torch.cat([self.transform(Image.open(s)) for s in samples][1:], dim=0)
        x_cond = torch.cat([self.downsample_tfm(Image.open(s)) for s in samples][1:], dim=0)
        ### apply noise on x_cond
        cond_noise = torch.randn_like(x_cond) * 0.2
        x_cond = x_cond + cond_noise
        task = self.tasks[idx]
        return x, x_cond, task
    
class MySeqDatasetMW(SequentialDataset):
    def __init__(self, path="../datasets/dataset_0513", sample_per_seq=8, target_size=(64, 64)):
        print("Preparing dataset...")
        self.sample_per_seq = sample_per_seq

        sequence_dirs = glob(f"{path}/**/metaworld_dataset/*/*/", recursive=True)
        self.tasks = []
        self.sequences = []
        for seq_dir in sequence_dirs:
            seq = self.get_samples(sorted(glob(f"{seq_dir}*")))
            self.sequences.append(seq)
            self.tasks.append(seq_dir.split("/")[-3].replace("-", " "))
        
        
        self.transform = T.Compose([
            T.CenterCrop((128, 128)),
            T.Resize(target_size),
            T.ToTensor()
        ])
        print("Done")

### Randomly sample, from any intermediate to the last frame
# included_tasks = ["door-open", "door-close", "basketball", "shelf-place", "button-press", "button-press-topdown", "faucet-close", "faucet-open", "handle-press", "hammer", "assembly"]
# included_idx = [i for i in range(5)]
class SequentialDatasetv2(Dataset):
    def __init__(self, path="../datasets/valid", sample_per_seq=7, target_size=(128, 128), frameskip=None, randomcrop=False):
        print("Preparing dataset...")
        self.sample_per_seq = sample_per_seq

        self.frame_skip = frameskip

        sequence_dirs = glob(f"{path}/**/metaworld_dataset/*/*/*/", recursive=True)
        self.tasks = []
        self.sequences = []
        for seq_dir in sequence_dirs:
            task = seq_dir.split("/")[-4]
            seq_id= int(seq_dir.split("/")[-2])
            # if task not in included_tasks or seq_id not in included_idx:
            #     continue
            seq = sorted(glob(f"{seq_dir}*.png"), key=lambda x: int(x.split("/")[-1].rstrip(".png")))
            self.sequences.append(seq)
            self.tasks.append(seq_dir.split("/")[-4].replace("-", " "))
    
        if randomcrop:
            self.transform = video_transforms.Compose([
                video_transforms.CenterCrop((160, 160)),
                video_transforms.RandomCrop((128, 128)),
                video_transforms.Resize(target_size),
                volume_transforms.ClipToTensor()
            ])
        else:
            self.transform = video_transforms.Compose([
                video_transforms.CenterCrop((128, 128)),
                video_transforms.Resize(target_size),
                volume_transforms.ClipToTensor()
            ])
        print("Done")

    def get_samples(self, idx):
        seq = self.sequences[idx]
        # if frameskip is not given, do uniform sampling betweeen a random frame and the last frame
        if self.frame_skip is None:
            start_idx = random.randint(0, len(seq)-1)
            seq = seq[start_idx:]
            N = len(seq)
            samples = []
            for i in range(self.sample_per_seq-1):
                samples.append(int(i*(N-1)/(self.sample_per_seq-1)))
            samples.append(N-1)
        else:
            start_idx = random.randint(0, len(seq)-1)
            samples = [i if i < len(seq) else -1 for i in range(start_idx, start_idx+self.frame_skip*self.sample_per_seq, self.frame_skip)]
        return [seq[i] for i in samples]
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        try:
            samples = self.get_samples(idx)
            images = self.transform([Image.open(s) for s in samples]) # [c f h w]
            x_cond = images[:, 0] # first frame
            x = rearrange(images[:, 1:], "c f h w -> (f c) h w") # all other frames
            task = self.tasks[idx]
            return x, x_cond, task
        except Exception as e:
            print(e)
            return self.__getitem__(idx + 1 % self.__len__()) 
        
class SequentialFlowDataset(Dataset):
    def __init__(self, path="../datasets/valid", sample_per_seq=7, target_size=(128, 128), frameskip=None, randomcrop=False):
        print("Preparing dataset...")
        self.sample_per_seq = sample_per_seq

        self.frame_skip = frameskip

        sequence_dirs = glob(f"{path}/**/metaworld_dataset/*/*/*/", recursive=True)
        self.tasks = []
        self.sequences = []
        self.flows = []
        for seq_dir in sequence_dirs:
            task = seq_dir.split("/")[-4]
            seq_id= int(seq_dir.split("/")[-2])
            # if task not in included_tasks or seq_id not in included_idx:
            #     continue
            seq = sorted(glob(f"{seq_dir}*.png"), key=lambda x: int(x.split("/")[-1].rstrip(".png")))
            flows = sorted(glob(f"{seq_dir}flow/*.npy"))
            self.sequences.append(seq)
            self.flows.append(np.array([np.load(flow) for flow in flows]))
            self.tasks.append(seq_dir.split("/")[-4].replace("-", " "))

        self.transform = T.Compose([
            T.CenterCrop((128, 128)),
            T.Resize(target_size),
            T.ToTensor()
        ])
        
        print("Done")

    def get_samples(self, idx):
        seq = self.sequences[idx]
        return seq[0]
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # try:
            s = self.get_samples(idx)
            x_cond = self.transform(Image.open(s)) # [c f h w]
            x = rearrange(torch.from_numpy(self.flows[idx]), "f w h c -> (f c) w h") / 128
            task = self.tasks[idx]
            return x, x_cond, task
        # except Exception as e:
        #     print(e)
        #     return self.__getitem__(idx + 1 % self.__len__()) 

class SequentialNavDataset(Dataset):
    def __init__(self, path="../datasets/valid", sample_per_seq=8, target_size=(64, 64)):
        print("Preparing dataset...")
        self.sample_per_seq = sample_per_seq

        sequence_dirs = glob(f"{path}/**/thor_dataset/*/*/", recursive=True)
        self.tasks = []
        self.sequences = []
        for seq_dir in sequence_dirs:
            task = seq_dir.split("/")[-3]
            seq = sorted(glob(f"{seq_dir}frames/*.png"), key=lambda x: int(x.split("/")[-1].rstrip(".png")))
            self.sequences.append(seq)
            self.tasks.append(task)

        self.transform = video_transforms.Compose([
            video_transforms.Resize(target_size),
            volume_transforms.ClipToTensor()
        ])

        num_seqs = len(self.sequences)
        num_frames = sum([len(seq) for seq in self.sequences])
        self.num_frames = num_frames
        self.frameid2seqid = [i for i, seq in enumerate(self.sequences) for _ in range(len(seq))]
        self.frameid2seq_subid = [f - self.frameid2seqid.index(self.frameid2seqid[f]) for f in range(num_frames)]

        print(f"Found {num_seqs} seqs, {num_frames} frames in total")
        print("Done")

    def get_samples(self, idx):
        seqid = self.frameid2seqid[idx]
        seq = self.sequences[seqid]
        start_idx = self.frameid2seq_subid[idx]
        
        samples = [i if i < len(seq) else -1 for i in range(start_idx, start_idx+self.sample_per_seq)]
        return [seq[i] for i in samples]
    
    def __len__(self):
        return self.num_frames
    
    def __getitem__(self, idx):
        samples = self.get_samples(idx)
        images = self.transform([Image.open(s) for s in samples]) # [c f h w]
        x_cond = images[:, 0] # first frame
        x = rearrange(images[:, 1:], "c f h w -> (f c) h w") # all other frames
        task = self.tasks[self.frameid2seqid[idx]]
        return x, x_cond, task

class MySeqDatasetReal(SequentialDataset):
    def __init__(self, path="../datasets/dataset_0606/processed_data", sample_per_seq=7, target_size=(48, 64)):
        print("Preparing dataset...")
        self.sample_per_seq = sample_per_seq

        sequence_dirs = glob(f"{path}/*/*/", recursive=True)
        print(f"found {len(sequence_dirs)} sequences")
        self.tasks = []
        self.sequences = []
        for seq_dir in sequence_dirs:
            seq = self.get_samples(sorted(glob(f"{seq_dir}*.png")))
            self.sequences.append(seq)
            self.tasks.append(seq_dir.split("/")[-3].replace("_", " "))
        
        self.transform = T.Compose([
            T.Resize(target_size),
            T.ToTensor()
        ])
        print("Done")


if __name__ == "__main__":
    dataset = SequentialNavDataset("../datasets/thor")
    x, x_cond, task = dataset[2]
    print(x.shape)
    print(x_cond.shape)
    print(task)

