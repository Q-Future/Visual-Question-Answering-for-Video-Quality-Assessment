<div align="center">
    <a href="https://huggingface.co/spaces/teowu/Q-Instruct-on-mPLUG-Owl-2"><img src="https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm-dark.svg" alt="Open in Spaces"></a>
<a href="https://arxiv.org/abs/2311.06783"><img src="https://img.shields.io/badge/Arxiv-2311:06783-red"></a>
    <a href="https://huggingface.co/datasets/teowu/Q-Instruct"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-green"></a>
    <a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FQ-Future%2FQ-Instruct&count_bg=%23E97EBA&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=visitors&edge_flat=false" alt="Hits"></a>
    <a href='https://github.com/Q-Future/Q-Instruct/stargazers'><img src='https://img.shields.io/github/stars/Q-Future/Q-Instruct.svg?style=social'></a>
</div>
# VQA²-Visual-Question-Answering-for-Video-Quality-Assessment
Official code and dataset for VQA² series models and dataset
## Quicker Start:
Install dependencies:
```shell
cd VQA_main
conda create -n VQA python=3.10 -y
conda activate VQA
pip install --upgrade pip  # Enable PEP 660 support.
pip install -e ".[train]"
pip install pytorchvideo #For slowfast base model download
pip install transformers==4.44.0 #Change the transformers version
```
Notice!
Replace the VQA/python3.10/site-packages/transformers/models/qwen2/modeling_qwen2.py to VQA_main/modeling_qwen2.py (we set some customized parameters in it).

## VQA² Scorers: 
cd VQA_main

For UGC-Video Scoring:
```shell
python ./llava/eval/model_score_UGC.py
```
For Streaming-Video Scoring:
```shell
python ./llava/eval/model_score_streaming.py
```
## VQA² Assistant: 
cd VQA_benchmark_test

For Q-bench-video Evaluation:
```shell
python ./llava/eval/model_vqa_q_bench_video.py
```
For Simple Q&A:
```shell
python ./llava/eval/model_conv.py
```
Gradio demo:
```shell
python ./app.py #Note that the minimux GPU requirement is 3090(24G)*1.
```
## Training
cd VQA_main
```shell
chmod +x ./scripts/train/finetune_VQA².sh
```
Then directly execute this .sh file. Note that we only support training with per_device_train_batch_size=1.

## Model Zoo
We temporarily provide the huggingface weight of VQA²-UGC-Scorer(7B) ,VQA²-Streaming-Scorer(7B), and VQA²-Assistant(7B); more versions will be released later.

HF-PATH:

VQA²-UGC-Scorer(7B): https://huggingface.co/q-future/VQA-UGC-Scorer (q-future/VQA-UGC-Scorer)

VQA²-Streaming-Scorer(7B): https://huggingface.co/q-future/VQA-Streaming-Scorer (q-future/VQA-Streaming-Scorer)

VQA²-Assistant(7B): https://huggingface.co/q-future/VQA-Assistant (q-future/VQA-Assistant)




