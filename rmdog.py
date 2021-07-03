import torch, os, pdb, collections, json, PIL, pickle
import numpy as np
from typing import Any, Callable, Optional, Tuple, List
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive, check_integrity
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from randaugment import RandAugment
from utils.augment import Cutout, select_autoaugment
from utils.data_loader import get_statistics


class DOG_DISJOINT(Dataset):
    def __init__(self,
        root = "./dataset/dog120",
        prefix = "collections/dog120/",
        mode=None,
        session_id = None,
        joint_train = False,
        seed = 1,
        exp_name = "disjoint",
    ):
        assert mode in ["train","gallery","val","test"], "mode should be {train, gallery, val, test}"
        datalist = []
        if mode in ["test"]:
            tmp = []
            for sid in range(4):
                collection_name = os.path.join(prefix,"dog120_test_rand1_cls30_task%d.json"%(sid))
                with open(collection_name,"r") as f: X = json.load(f)
                tmp.append(X)
            datalist = [item for sublist in tmp[:session_id+1] for item in sublist]
        else:#mode in ["gallery", "train", "val"]
            np.random.seed(seed)#for reproducibility
            X_train_list, X_val_list = [], []
            for sid in range(4):
                collection_name = os.path.join(prefix,"dog120_train_general30_rand1_cls20_task%d.json"%(sid))
                with open(collection_name,"r") as f: X = json.load(f)
                y = [item['label'] for item in X]
                #pdb.set_trace()
                X_train, X_val, _, _ = train_test_split(X, y,stratify=y, test_size=0.1,random_state=seed)# seed for reproducibility
                X_train_list.append(X_train)
                X_val_list.append(X_val)
            if mode in ["val"]:
                datalist = [item for sublist in X_val_list[:session_id+1] for item in sublist]
            else:#mode in ["gallery","train"]
                if joint_train:#collect train set seen so far
                    datalist = [item for sublist in X_train_list[:session_id+1] for item in sublist]
                else:#collect train set for assigned session
                    datalist = X_train_list[session_id]
        self.data = np.array([item["file_name"] for item in datalist])
        self.targets = [item["label"] for item in datalist]
        self.root = root
        self.mode = mode
        if mode == "train":
            self.desc = "training set"
        elif mode == "gallery":
            self.desc = "gallery set"
        elif mode == "val":
            self.desc = "validation query set"
        elif mode == "test":
            self.desc = "testing query set"
        # =================================================================================
        # Transform Definition
        crop_im_size = 224
        mean, std, _, _, _ = get_statistics(dataset='imagenet1000')
        if mode == "train":
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(size=crop_im_size),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
            print("Using train-transforms:",self.transform)
        else:#mode in ["gallery", "val", "test"]
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(crop_im_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.data[idx]
        label = self.targets[idx]
        img_path = os.path.join(self.root, img_name)
        imgpil = PIL.Image.open(img_path).convert("RGB")
        if self.transform:
            trans_img = self.transform(imgpil)
        # return based on self.desc
        if self.mode == "gallery":
            return trans_img, label, img_name
        else:
            return trans_img, label
    
    def add_memory(self, exem_data, exem_labels):
        exem_data = exem_data.tolist()
        exem_labels = exem_labels.tolist()
        tmp = self.data.tolist()
        tmp.extend(exem_data)
        self.data = np.array(tmp)
        self.targets.extend(exem_labels)
        print("##### [add memory] #####")
        cat_dict = {}
        for cat in exem_labels:
            if cat not in cat_dict:
                cat_dict[cat] = 0
            cat_dict[cat] += 1
        print(cat_dict)
        print("########################")
    
    def show(self,verbose=True):
        print("-----------------")
        print("[%s]"%(self.desc))
        print("class label from %d to %d"%(np.min(self.targets),np.max(self.targets)))
        print("number of data: ",self.data.shape," with dtype %s"%(self.data.dtype))
        if verbose: print({tar:cnt for tar, cnt in zip(*np.unique(self.targets, return_counts=True))})
        print("-----------------")
