import os
from torch.utils.data import Dataset
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import csv

from .self_sup_tasks import patch_ex



CLASS_NAMES = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2','pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']


describles = {}
describles['candle'] = "This is a photo of 4 candles for anomaly detection, every candle should be round, without any damage, flaw, defect, scratch, hole or broken part."
describles['capsules'] = "This is a photo of many small capsules for anomaly detection, every capsule is green, should be without any damage, flaw, defect, scratch, hole or broken part."
describles['cashew'] = "This is a photo of a cashew for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['chewinggum'] = "This is a photo of a chewinggom for anomaly detection, which should be white, without any damage, flaw, defect, scratch, hole or broken part."
describles['fryum'] = "This is a photo of a fryum for anomaly detection on green background, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['macaroni1'] = "This is a photo of 4 macaronis for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['macaroni2'] = "This is a photo of 4 macaronis for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['pcb1'] = "This is a photo of pcb for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['pcb2'] = "This is a photo of pcb for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['pcb3'] = "This is a photo of pcb for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['pcb4'] = "This is a photo of pcb for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['pipe_fryum'] = "This is a photo of a pipe fryum for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."


class VisaDataset(Dataset):
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.transform = transforms.Resize(
                                (224, 224), interpolation=transforms.InterpolationMode.BICUBIC
                            )
        
        self.norm_transform = transforms.Compose(
                            [
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    mean=(0.48145466, 0.4578275, 0.40821073),
                                    std=(0.26862954, 0.26130258, 0.27577711),
                                ),
                            ]
                        )
        
        datas_csv_path = '../data/VisA/split_csv/1cls.csv'

        self.paths = []
        self.x = []

        with open(datas_csv_path, 'r') as file:
            reader = csv.reader(file)

            for row in reader:
                if row[1] == 'train' and row[0] in CLASS_NAMES:
                    file_path = os.path.join(root_dir, row[3])
                    self.paths.append(file_path)
                    self.x.append(self.transform(Image.open(file_path).convert('RGB')))

        
        self.prev_idx = np.random.randint(len(self.paths))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):

        img_path, x = self.paths[index], self.x[index]
        class_name = img_path.split('/')[-5]

        self_sup_args={'width_bounds_pct': ((0.03, 0.4), (0.03, 0.4)),
                    'intensity_logistic_params': (1/12, 24),
                    'num_patches': 2,
                    'min_object_pct': 0,
                    'min_overlap_pct': 0.25,
                    'gamma_params':(2, 0.05, 0.03), 'resize':True, 
                    'shift':True, 
                    'same':False, 
                    'mode':cv2.NORMAL_CLONE, 
                    'label_mode':'logistic-intensity',
                    'skip_background': None,
                    'resize_bounds': (.5, 2)
                    }

        x = np.asarray(x)
        origin = x

        p = self.x[self.prev_idx]
        if self.transform is not None:
            p = self.transform(p)
        p = np.asarray(p)    
        x, mask, centers = patch_ex(x, p, **self_sup_args)
        mask = torch.tensor(mask[None, ..., 0]).float()
        self.prev_idx = index
        

        origin = self.norm_transform(origin)
        x = self.norm_transform(x)

   
        if len(centers) > 0:
            position = []
            for center in centers:
                center_x = center[0] / 224
                center_y = center[1] / 224

                if center_x <= 1/3 and center_y <= 1/3:
                    position.append('top left')
                elif center_x <= 1/3 and center_y > 1/3 and center_y <= 2/3:
                    position.append('top')
                elif center_x <= 1/3 and center_y > 2/3:
                    position.append('top right')

                elif center_x <= 2/3 and center_y <= 1/3:
                    position.append('left')
                elif center_x <= 2/3 and center_y > 1/3 and center_y <= 2/3:
                    position.append('center')
                elif center_x <= 2/3 and center_y > 2/3:
                    position.append('right')

                elif center_y <= 1/3:
                    position.append('bottom left')
                elif center_y > 1/3 and center_y <= 2/3:
                    position.append('bottom')
                elif center_y > 2/3:
                    position.append('bottom right')

            conversation_normal = []

            conversation_normal.append({"from":"human","value": describles[class_name] + " Is there any anomaly in the image?"})
            conversation_normal.append({"from":"gpt","value":"No, there is no anomaly in the image."})

            conversation_abnormal = []
            conversation_abnormal.append({"from":"human","value":  describles[class_name] + " Is there any anomaly in the image?"})


            

            if len(centers) > 1:
                abnormal_describe =  "Yes, there are " + str(len(centers)) + " anomalies in the image, they are at the "
                for i in range(len(centers)):
                    if i == 0:
                        abnormal_describe += position[i]

                    elif i == 1 and position[i] != position[i-1]:
                        if i != len(centers) - 1:
                            abnormal_describe += ", "
                            abnormal_describe += position[i]
                        else:
                            abnormal_describe += " and " + position[i] + " of the image."
                    
                    elif i == 1 and position[i] == position[i-1]:
                        if i == len(centers) - 1:
                            abnormal_describe += " of the image."

            else:
                abnormal_describe = "Yes, there is an anomaly in the image, at the " + position[0] + " of the image."

            conversation_abnormal.append({"from":"gpt","value":abnormal_describe})

        else:
            print("no mask")
            conversation_normal = []

            conversation_normal.append({"from":"human","value": describles[class_name] + " Is there any anomaly in the image?"})
            conversation_normal.append({"from":"gpt","value":"No, there is no anomaly in the image."})


            conversation_abnormal = conversation_normal


        return origin, conversation_normal, x, conversation_abnormal, class_name, mask, img_path



    def collate(self, instances):

        images = []
        texts = []
        class_names = []
        masks = []
        img_paths = []
        for instance in instances:
            images.append(instance[0])
            texts.append(instance[1])
            class_names.append(instance[4])
            masks.append(torch.zeros_like(instance[5]))
            img_paths.append(instance[6])

            images.append(instance[2])
            texts.append(instance[3])
            class_names.append(instance[4])
            masks.append(instance[5])
            img_paths.append(instance[6])


        return dict(
            images=images,
            texts=texts,
            class_names=class_names,
            masks=masks,
            img_paths=img_paths
        )