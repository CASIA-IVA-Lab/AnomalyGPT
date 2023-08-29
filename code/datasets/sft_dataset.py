#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import os
import json
from tqdm import tqdm
import ipdb
import random
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass, field
from typing import Callable, Dict, Sequence

import torch
import torch.distributed as dist
import transformers
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, image_root_path: str):
        super(SupervisedDataset, self).__init__()

        with open(data_path, 'r') as f:
            json_data = json.load(f)
            # for debug:
            # json_data = json_data[:10000]

        self.norm_transform = transforms.Compose(
                            [
                                transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    mean=(0.48145466, 0.4578275, 0.40821073),
                                    std=(0.26862954, 0.26130258, 0.27577711),
                                ),
                            ]
                        )

        self.image_path_list, self.caption_list = [], []
        for item in json_data:
            one_image_name, one_caption = item["image_name"], item["conversation"]
            if len(one_caption) > 2:
                one_caption = one_caption[:2]
            # print(one_caption)
            # TODO: stage 2 dataset format is invalid
            if not one_image_name.endswith('.jpg'):
                one_image_name += '.jpg'
            one_image_path = image_root_path + '/{}'.format(one_image_name)
            self.image_path_list.append(one_image_path)
            self.caption_list.append(one_caption)
        print(f'[!] collect {len(self.image_path_list)} samples for training')

    def __len__(self): # number of instances
        return len(self.image_path_list)

    #def __getitem__(self, i) -> Dict[str, torch.Tensor]: # how to get item, 取一个样本
    def __getitem__(self, i):
        texts = self.caption_list[i]
        print(texts)
        image_path = self.image_path_list[i]
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.norm_transform(image)
        return dict(image_paths = image_tensor, output_texts=texts)

    def collate(self, instances):
        image_paths, output_texts = tuple([instance[key] for instance in instances] for key in ("image_paths", "output_texts"))
        return dict(
            image_paths=image_paths,
            output_texts=output_texts
        )
