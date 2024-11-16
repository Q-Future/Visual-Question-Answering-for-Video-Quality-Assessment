# VQA²-Visual-Question-Answering-for-Video-Quality-Assessment
Official code and dataset for VQA² series models and dataset
## Quicker Start:
Install dependencies:
```shell
cd VQA²-main
conda create -n VQA python=3.10 -y
conda activate VQA
pip install --upgrade pip  # Enable PEP 660 support.
pip install -e ".[train]"
pip install pytorchvideo #For slowfast base model download
pip install transformers==4.44.0 #Change the transformers version
```
Notice!
Replace the VQA/python3.10/site-packages/transformers/models/qwen2/modeling_qwen2.py to VQA²-main/modeling_qwen2.py （we set some customized parameters in it）

## VQA² Scorers: 
cd VQA²-main

For UGC-Video Scoring:
```shell
python ./llava/eval/model_score_UGC.py
For Streaming-Video Scorering:
python ./llava/eval/model_score_streaming.py
```
## VQA² Assistant: 
cd VQA²_benchmark

For Q-bench-video Evaluation:
```shell
cd VQA²_benchmark
python ./llava/eval/model_vqa_q_bench_video.py
```
For simple Q&A:
```shell
python ./llava/eval/model_conv.py
```
Gradio demo:
```shell
python ./app.py #Note that the minimux GPU requirement is 3090(24G)*1.
```


