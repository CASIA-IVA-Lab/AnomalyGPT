# 1. Prepare Vicuna Checkpoint:

The language decoder of AnomalyGPT is based on Vicuna version 0. Given the distribution license of LLaMA, you need to restore the weights of Vicuna manually. To restore the weights, please follow the instructions below. In the following, we showcase how to restore the 7B version of Vicuna v0. To obtain the 13B version of Vicuna, you can take similar procedures. 

## 1.1. Obtain LLaMA Weights:
* Request the weights of LLaMA from Meta using [this form](https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform).
* After obtaining the weights of a specific LLaMA (e.g. 7B, 13B), following [instructions](https://huggingface.co/docs/transformers/main/model_doc/llama) provided by Huggingface to convert it into Huggingface format. 

> **** After conversion, the directory should look like:

    .
    └── ./{path_to_llama_weights}/             
        ├── config.json
        ├── generation_config.json
        ├── pytorch_model-00001-of-00002.bin
        ├── pytorch_model-00002-of-00002.bin
        ├── pytorch_model.bin.index.json
        ├── special_tokens_map.json
        ├── tokenizer.model
        └── tokenizer_config.json
        
`{path_to_llama_weights}` is where you store the checkpoints.


## 1.2. Obtain the Delta Weights of Vicuna:

Then, you should download the delta weights of Vicuna provided by the original authors. You can find the corresponding links to 7B/13B Vicuna models in the table below.

|**Model Size**|**Delta Weights Address**|**Version**|
|:-------------:|:-------------:|:-------------:|
|7B|[[Link]](https://huggingface.co/lmsys/vicuna-7b-delta-v0)|0|
|13B|[[Link]](https://huggingface.co/lmsys/vicuna-13b-delta-v0)|0|



> **** After conversion, the directory should look like:

    .
    └── ./{path_to_delta_vicuna_weights}/             
        ├── config.json
        ├── generation_config.json
        ├── pytorch_model-00001-of-00002.bin
        ├── pytorch_model-00002-of-00002.bin
        ├── pytorch_model.bin.index.json
        ├── special_tokens_map.json
        ├── tokenizer.model
        └── tokenizer_config.json
      
`{path_to_delta_vicuna_weights}` is where you store the delta weights of Vicuna.

## 1.3. Combine the Weights:

When the two sets of weights are ready, you can combine them using tools from the Vicuna team.

First, install the required library.
```yaml
pip install git+https://github.com/lm-sys/FastChat.git@v0.1.10
```

Then, run the following command. 
```yaml
python -m fastchat.model.apply_delta --base {path_to_llama_weights}  --target ./vicuna_ckpt/7b_v0/  --delta {path_to_delta_vicuna_weights}
```

> **** Now, the final weights are ready as:

    .
    └── ./vicuna_ckpt/7b_v0/             
        ├── config.json
        ├── generation_config.json
        ├── pytorch_model-00001-of-00002.bin
        ├── pytorch_model-00002-of-00002.bin
        ├── pytorch_model.bin.index.json
        ├── special_tokens_map.json
        ├── tokenizer.model
        └── tokenizer_config.json


