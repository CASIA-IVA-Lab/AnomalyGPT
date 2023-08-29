import os
from model.openllama import OpenLLAMAPEFTModel
import torch
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from PIL import Image
import numpy as np
import csv
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser("AnomalyGPT", add_help=True)
# paths
parser.add_argument("--few_shot", type=bool, default=True)
parser.add_argument("--k_shot", type=int, default=1)
parser.add_argument("--round", type=int, default=14)


command_args = parser.parse_args()


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

FEW_SHOT = command_args.few_shot 

# init the model
args = {
    'model': 'openllama_peft',
    'imagebind_ckpt_path': '../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth',
    'vicuna_ckpt_path': '../pretrained_ckpt/vicuna_ckpt/7b_v0',
    'anomalygpt_ckpt_path': './ckpt/train_mvtec/pytorch_model.pt',
    'delta_ckpt_path': '../pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt',
    'stage': 2,
    'max_tgt_len': 128,
    'lora_r': 32,
    'lora_alpha': 32,
    'lora_dropout': 0.1,
}

model = OpenLLAMAPEFTModel(**args)
delta_ckpt = torch.load(args['delta_ckpt_path'], map_location=torch.device('cpu'))
model.load_state_dict(delta_ckpt, strict=False)
delta_ckpt = torch.load(args['anomalygpt_ckpt_path'], map_location=torch.device('cpu'))
model.load_state_dict(delta_ckpt, strict=False)
model = model.eval().half().cuda()

print(f'[!] init the 7b model over ...')

"""Override Chatbot.postprocess"""


def predict(
    input, 
    image_path,
    normal_img_path, 
    max_length, 
    top_p, 
    temperature,
    history,
    modality_cache,  
):
    
    prompt_text = ''
    for idx, (q, a) in enumerate(history):
        if idx == 0:
            prompt_text += f'{q}\n### Assistant: {a}\n###'
        else:
            prompt_text += f' Human: {q}\n### Assistant: {a}\n###'
    if len(history) == 0:
        prompt_text += f'{input}'
    else:
        prompt_text += f' Human: {input}'

    response, pixel_output = model.generate({
        'prompt': prompt_text,
        'image_paths': [image_path] if image_path else [],
        'audio_paths': [],
        'video_paths': [],
        'thermal_paths': [],
        'normal_img_paths': normal_img_path if normal_img_path else [],
        'top_p': top_p,
        'temperature': temperature,
        'max_tgt_len': max_length,
        'modality_embeds': modality_cache
    })

    return response, pixel_output

input = "Is there any anomaly in the image?"

root_dir = '../data/VisA'

mask_transform = transforms.Compose([
                                transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()
                            ])

datas_csv_path = '../data/VisA/split_csv/1cls.csv'



CLASS_NAMES = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2','pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']

precision = []
file_paths = {}
normal_img_path = {}

for class_name in CLASS_NAMES:
    file_paths[class_name] = []
    normal_img_path[class_name] = []

with open(datas_csv_path, 'r') as file:
    reader = csv.reader(file)

    for row in reader:
        if row[1] == 'test' and row[0] in CLASS_NAMES:
            file_paths[row[0]].append(os.path.join(root_dir, row[3]))
        if row[0] in CLASS_NAMES and len(normal_img_path[row[0]]) < command_args.round * 4 + command_args.k_shot and row[1] == 'train':
            normal_img_path[row[0]].append(os.path.join(root_dir, row[3]))


if FEW_SHOT:
    for i in CLASS_NAMES:
        normal_img_path[i] = normal_img_path[i][command_args.round * 4:]


p_auc_list = []
i_auc_list = []

for c_name in CLASS_NAMES:
    right = 0
    wrong = 0
    p_pred = []
    p_label = []
    i_pred = []
    i_label = []
    for file_path in tqdm(file_paths[c_name]):
        if FEW_SHOT:
            resp, anomaly_map = predict(describles[c_name] + ' ' + input, file_path, normal_img_path[c_name], 512, 0.01, 1.0, [], [])
        else:
            resp, anomaly_map = predict(describles[c_name] + ' ' + input, file_path, None, 512, 0.01, 1.0, [], [])
        is_normal = 'Normal' in file_path.split('/')[-2]

        if is_normal:
            img_mask = Image.fromarray(np.zeros((224, 224)), mode='L')
        else:
            mask_path = file_path.replace('Images', 'Masks')
            mask_path = mask_path.replace('.JPG', '.png')
            img_mask = Image.open(mask_path).convert('L')

        img_mask = mask_transform(img_mask)
        threshold = img_mask.max() / 100
        img_mask[img_mask > threshold], img_mask[img_mask <= threshold] = 1, 0
        img_mask = img_mask.squeeze().reshape(224, 224).cpu().numpy()
        
        anomaly_map = anomaly_map.reshape(224, 224).detach().cpu().numpy()

        p_label.append(img_mask)
        p_pred.append(anomaly_map)

        i_label.append(1 if not is_normal else 0)
        i_pred.append(anomaly_map.max())


        # print(file_path, resp)
        if 'Normal' not in file_path and 'Yes' in resp:
            right += 1
        elif 'Normal' in file_path and 'No' in resp:
            right += 1
        else:
            wrong += 1

    p_pred = np.array(p_pred)
    p_label = np.array(p_label)

    i_pred = np.array(i_pred)
    i_label = np.array(i_label)

    p_auroc = round(roc_auc_score(p_label.ravel(), p_pred.ravel()) * 100,2)
    i_auroc = round(roc_auc_score(i_label.ravel(), i_pred.ravel()) * 100,2)
    
    p_auc_list.append(p_auroc)
    i_auc_list.append(i_auroc)
    precision.append(100 * right / (right + wrong))

    print(c_name, 'right:',right,'wrong:',wrong)
    print(c_name, "i_AUROC:", i_auroc)
    print(c_name, "p_AUROC:", p_auroc)

print("i_AUROC:",torch.tensor(i_auc_list).mean())
print("p_AUROC:",torch.tensor(p_auc_list).mean())
print("precision:",torch.tensor(precision).mean())
