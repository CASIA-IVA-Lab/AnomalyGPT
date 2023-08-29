import gradio as gr
import mdtex2html
from model.openllama import OpenLLAMAPEFTModel
import torch
from io import BytesIO
from PIL import Image as PILImage
import cv2
import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms

# init the model
args = {
    'model': 'openllama_peft',
    'imagebind_ckpt_path': '../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth',
    'vicuna_ckpt_path': '../pretrained_ckpt/vicuna_ckpt/7b_v0',
    'anomalygpt_ckpt_path': './ckpt/train_cn/pytorch_model.pt',
    'delta_ckpt_path': '../pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt',
    'stage': 2,
    'max_tgt_len': 128,
    'lora_r': 32,
    'lora_alpha': 32,
    'lora_dropout': 0.1,
    'layers': [7,15,23,31]
}

model = OpenLLAMAPEFTModel(**args)
delta_ckpt = torch.load(args['delta_ckpt_path'], map_location=torch.device('cpu'))
model.load_state_dict(delta_ckpt, strict=False)
delta_ckpt = torch.load(args['anomalygpt_ckpt_path'], map_location=torch.device('cpu'))
model.load_state_dict(delta_ckpt, strict=False)
model = model.eval().half().cuda()

output = None

"""Override Chatbot.postprocess"""
def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text


def predict(
    input, 
    image_path, 
    normal_img_path,  
    chatbot, 
    max_length, 
    top_p, 
    temperature, 
    history, 
    modality_cache, 
):
    
    if image_path is None and normal_img_path is None:
        return [(input, "There is no input data provided! Please upload your data and start the conversation.")]
    else:
        print(f'[!] image path: {image_path}\n[!] normal image path: {normal_img_path}\n')

    # prepare the prompt
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
        'normal_img_paths': [normal_img_path] if normal_img_path else [],
        'audio_paths': [],
        'video_paths': [],
        'thermal_paths': [],
        'top_p': top_p,
        'temperature': temperature,
        'max_tgt_len': max_length,
        'modality_embeds': modality_cache
    },web_demo=True)
    chatbot.append((parse_text(input), parse_text(response)))
    history.append((input, response))


    plt.imshow(pixel_output.reshape(224,224).detach().cpu(), cmap='binary_r')
    plt.axis('off')
    plt.savefig('output.png',bbox_inches='tight',pad_inches = 0)

    target_size = 224
    original_width, original_height = PILImage.open(image_path).size
    if original_width > original_height:
        new_width = target_size
        new_height = int(target_size * (original_height / original_width))
    else:
        new_height = target_size
        new_width = int(target_size * (original_width / original_height))

    new_image = PILImage.new('L', (target_size, target_size), 255)  # 'L' mode for grayscale

    paste_x = (target_size - new_width) // 2
    paste_y = (target_size - new_height) // 2

    pixel_output = PILImage.open('output.png').resize((new_width, new_height), PILImage.LANCZOS)

    new_image.paste(pixel_output, (paste_x, paste_y))

    new_image.save('output.png')

    image = cv2.imread('output.png', cv2.IMREAD_GRAYSCALE)
    kernel = np.ones((3, 3), np.uint8)
    eroded_image = cv2.erode(image, kernel, iterations=1)
    cv2.imwrite('output.png', eroded_image)

    global output
    output =  PILImage.open('output.png').convert('L')


    return chatbot, history, modality_cache


def get_image():
    global output
    return output if output else "ffffff.png"


def reset_user_input():
    return gr.update(value='')

def reset_dialog():
    return [], []

def reset_state():
    global output
    output = None
    return None, None, [], [], []



with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">Demo of AnomalyGPT</h1>""")

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row(scale=3):
                image_path = gr.Image(type="filepath", label="Query Image", value=None)
            with gr.Row(scale=3):
                normal_img_path = gr.Image(type="filepath", label="Normal Image", value=None)
            with gr.Row():
                max_length = gr.Slider(0, 512, value=512, step=1.0, label="Maximum length", interactive=True)
            with gr.Row():
                top_p = gr.Slider(0, 1, value=0.01, step=0.01, label="Top P", interactive=True)
            with gr.Row():
                temperature = gr.Slider(0, 1, value=1.0, step=0.01, label="Temperature", interactive=True)


        with gr.Column(scale=3):
            with gr.Row():
                with gr.Column(scale=6):
                    chatbot = gr.Chatbot().style(height=415)
                with gr.Column(scale=4):
                    # gr.Image(output)
                    image_output = gr.Image(value=get_image, label="Localization Output", every=1.0, shape=[224,224])
            with gr.Row():
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(container=False)
            with gr.Row():
                with gr.Column(scale=2):
                    submitBtn = gr.Button("Submit", variant="primary")
                with gr.Column(scale=1):
                    emptyBtn = gr.Button("Clear History")
                    
    history = gr.State([])
    modality_cache = gr.State([])

    submitBtn.click(
        predict, [
            user_input, 
            image_path, 
            normal_img_path, 
            chatbot, 
            max_length, 
            top_p, 
            temperature, 
            history, 
            modality_cache,
        ], [
            chatbot, 
            history,
            modality_cache
        ],
        show_progress=True
    )

    submitBtn.click(reset_user_input, [], [user_input])
    emptyBtn.click(reset_state, outputs=[
        image_path,
        normal_img_path,
        chatbot, 
        history, 
        modality_cache
    ], show_progress=True)


demo.queue().launch(server_port=24008)
