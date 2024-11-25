<div align="center">
  
<a href="https://arxiv.org/abs/2411.03795"><img src="https://img.shields.io/badge/Arxiv-2411:03795-red"></a>
  
</div>
</div>
    
  <h1> VQA²-Visual-Question-Answering-for-Video-Quality-Assessment</h1>

<div>

Official code and dataset for VQA² series models and dataset

## Exellent Performance on Video Quality Scoring and Video Quality Understanding!!!
 <div style="width: 100%; text-align: center; margin:auto;">
      <img style="width:100%" src="performance_00.png">
  </div>

## Dataset Construction Pipeline:
<div style="width: 100%; text-align: center; margin:auto;">
      <img style="width:100%" src="pipeline_00.png">
  </div>

## Model Structure:
<div style="width: 100%; text-align: center; margin:auto;">
      <img style="width:100%" src="model_00.png">
  </div>

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

## Citation

If you consider this work interesting, please feel free to cite it in your work!

```bibtex
@misc{Jia2024VQA,
      title={VQA²-Visual-Question-Answering-for-Video-Quality-Assessment}, 
      author={Ziheng Jia and Zicheng Zhang and Jiaying Qian and Haoning Wu and Wei Sun and Chunyi Li and Xiaohong Liu and Weisi Lin and Guangtao Zhai and Xiongkuo Min},
      year={2024},
      eprint={2411.03795},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```



