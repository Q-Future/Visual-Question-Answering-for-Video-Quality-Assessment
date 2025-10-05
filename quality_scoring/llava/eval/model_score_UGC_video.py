import argparse

import os
import sys
print(sys.path)
sys.path.append('./')
import json
from tqdm import tqdm
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
os.environ['HF_ENDPOINT']= 'https://hf-mirror.com'
import torch
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from torchvision import transforms
from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from typing import Dict, Optional, Sequence, List
import transformers
import re
from collections import defaultdict
from PIL import Image
import math
from scipy.stats import spearmanr, pearsonr
# from torchprofile import profile_macs
# from thop import profile

def wa5(logits):
    import numpy as np
    logprobs = np.array([logits["high"], logits["good"], logits["fair"], logits["poor"], logits["low"]])
    probs = np.exp(logprobs) / np.sum(np.exp(logprobs))
    return np.inner(probs, np.array([1,0.8,0.6,0.4,0.2]))

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

# def load_video(video_file,frames_limit):
#     from decord import VideoReader
#     vr = VideoReader(video_file)
#
#     # Get video frame rate
#     fps = vr.get_avg_fps()
#
#     # Calculate frame indices for 1fps
#     frame_indices = [int(len(vr) /frames_limit)* i for i in range(frames_limit)]
#     if len(frame_indices)==0:
#         frame_indices=[0]
#     frames = vr.get_batch(frame_indices).asnumpy()
#     return [Image.fromarray(frames[i]) for i in range(len(frame_indices))]
def load_video(video_file, video_fps):
    from decord import VideoReader,cpu
    vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
    # frame_idx=[]
    # for ii in range(len(vr)//round(vr.get_avg_fps())):
    #     total_frame_num = round(vr.get_avg_fps())
    #     avg_fps = round(vr.get_avg_fps() / video_fps)
    #     # total_frame_num=len(vr)//avg_fps*avg_fps
    #     frame_idx.extend([i for i in range(ii*round(vr.get_avg_fps()), (ii+1)*round(vr.get_avg_fps()), avg_fps)])
    # total_frame_num = len(vr)-(len(vr)//round(vr.get_avg_fps())*round(vr.get_avg_fps()))
    # avg_fps = round(vr.get_avg_fps() / video_fps)
    # # total_frame_num=len(vr)//avg_fps*avg_fps
    # frame_idx.extend([i for i in range((ii+1)*round(vr.get_avg_fps()), len(vr), avg_fps)])
    # if len(frame_idx) > 200:
    #             uniform_sampled_frames = np.linspace(0, total_frame_num - 1, 100, dtype=int)
    #             frame_idx = uniform_sampled_frames.tolist()
    frames = vr.get_batch(list(range(len(vr)))).asnumpy()
    frame_idx1 = []
    video_fps=1
    for ii in range(len(vr)//round(vr.get_avg_fps())):
        # print(video_file)
        total_frame_num = round(vr.get_avg_fps())
        avg_fps = round(vr.get_avg_fps() / video_fps)
        # total_frame_num=len(vr)//avg_fps*avg_fps
        frame_idx1.extend([i for i in range(ii*round(vr.get_avg_fps()), (ii+1)*round(vr.get_avg_fps()), avg_fps)])
    total_frame_num = len(vr)-(len(vr)//round(vr.get_avg_fps())*round(vr.get_avg_fps()))
    avg_fps = round(vr.get_avg_fps() / video_fps)
    # total_frame_num=len(vr)//avg_fps*avg_fps
    frame_idx1.extend([i for i in range((ii+1)*round(vr.get_avg_fps()), len(vr), avg_fps)])
    # if len(frame_idx) > 200:
    #             uniform_sampled_frames = np.linspace(0, total_frame_num - 1, 100, dtype=int)
    #             frame_idx = uniform_sampled_frames.tolist()

    return [Image.fromarray(frames[i]) for i in range(len((vr)))],frame_idx1
    # return frame_idx,len(frame_idx)/video_fps

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}

    im_start, im_end = tokenizer.additional_special_tokens_ids
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []

    source = sources
    # if roles[source[0]["from"]] != roles["human"]:
    #     source = source[1:]

    input_id, target = [], []
    system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
    input_id += system
    target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
    assert len(input_id) == len(target)
    for j, sentence in enumerate(source):
        if j==0:
            role = "<|im_start|>user"
        else:
            role = "<|im_start|>assistant"
        if has_image and sentence is not None and "<image>" in sentence:
            num_image = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence))
            texts = sentence.split('<image>')
            _input_id = tokenizer(role).input_ids + nl_tokens
            for i,text in enumerate(texts):
                _input_id += tokenizer(text).input_ids
                if i<len(texts)-1:
                    _input_id += [IMAGE_TOKEN_INDEX]
            _input_id += [im_end] + nl_tokens
            assert sum([i==IMAGE_TOKEN_INDEX for i in _input_id])==num_image
        else:
            if sentence["value"] is None:
                _input_id = tokenizer(role).input_ids + nl_tokens
            else:
                _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
        input_id += _input_id
        if role == "<|im_start|>user":
            _target = [im_start] + [IGNORE_INDEX] * (len(_input_id) - 3) + [im_end] + nl_tokens
        elif role == "<|im_start|>assistant":
            _target = [im_start] + [IGNORE_INDEX] * len(tokenizer(role).input_ids) + _input_id[len(tokenizer(role).input_ids) + 1 : -2] + [im_end] + nl_tokens
        else:
            raise NotImplementedError
        target += _target

    input_ids.append(input_id)
    targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    return input_ids

def eval_model(args):

    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name,attn_implementation=None)
    # model.save_pretrained("/tos-bjml-researcheval/jiaziheng/VQA++/Visual-Question-Answering-for-Video-Quality-Assessment/VQA_main/llava-ov-chat-qwen2_slowfast")
    model.half()
    os.makedirs(f"/tos-bjml-researcheval/jiaziheng/VQA++/Visual-Question-Answering-for-Video-Quality-Assessment/VQA_main/results/{args.model_path.split('/')[-1]}/", exist_ok=True)
    image_paths = [
        # "/DATA/DATA1/jzh/LIVE Video Quality Challenge (VQC) Database/Video/"
        # "/DATA/DATA1/jzh/konvid1k/",
        # "/DATA/DATA1/jzh/waterloo-IV/video",
        # "/DATA/DATA1/jzh/waterloo-IV/video",
        # "/DATA/DATA1/jzh/waterloo-IV/video",
        # "/DATA/DATA1/jzh/waterloo-IV/video",
        # "/DATA/DATA1/jzh/waterloo-IV/video"
        # "/fs-computility/mllm1/zhangzicheng/VQA++/sample2/",
        # "/tos-bjml-researcheval/jiaziheng/VQA++/konvid1k/",
        # "/tos-bjml-researcheval/jiaziheng/VQA++/LIVE-VQC/",
        "/tos-bjml-researcheval/jiaziheng/VQA++/YT-UGC/",
        # "/fs-computility/mllm1/zhangzicheng/VQA++/LSVQ/",
        "/tos-bjml-researcheval/jiaziheng/VQA++/LIVE-YT-Gaming",
        # "/tos-bjml-researcheval/jiaziheng/VQA++/konvid1k/",
        "/tos-bjml-researcheval/jiaziheng/VQA++/LSVQ/",
        "/tos-bjml-researcheval/jiaziheng/VQA++/LSVQ/",
        # "/tos-bjml-researcheval/jiaziheng/VQA++/LIVE-VQC/",
        # "/fs-computility/mllm1/zhangzicheng/VQA++/sample2/",

    ]

    json_prefix = '/tos-bjml-researcheval/jiaziheng/VQA++/Visual-Question-Answering-for-Video-Quality-Assessment/VQA_main/llava/eval/rating/'
    jsons = [
        # json_prefix + "UGC_test.json",
        # json_prefix + "LIVE-VQC.json",
        # json_prefix + "YT_UGC1.json",
        # # json_prefix + "konvid.json",
        # json_prefix + "LSVQ(test).json",
        # json_prefix + "konvid1k.json",
        # json_prefix + "LIVE-VQC.json",
        json_prefix + "YT_UGC1.json",
        json_prefix + "YT-Gaming.json",
        json_prefix + "lsvq1080p.json",
        json_prefix + "LSVQ(test).json",
        # json_prefix + "LIVE-VQC.json",
        # json_prefix + "UGC_test.json",
        # LSVQ(test).json
        # json_prefix + "waterloo_IV_round_1_test_ver1.json",
        # json_prefix + "waterloo_IV_round_2_test_ver1.json",
        # json_prefix + "waterloo_IV_round_3_test_ver1.json",
        # json_prefix + "waterloo_IV_round_4_test_ver1.json",
        # json_prefix + "waterloo_IV_round_5_test_ver1.json"
    ]
    spearmanr1 = []
    personr1 = []
    iqadata1=[]
    for image_path, json_ in zip(image_paths, jsons):
        spearmanr1 = []
        personr1 = []
        mse1 = []
        with open(json_) as f:
            iqadata = json.load(f)
            # print(len(iqadata))
    #         iqadata1.extend(iqadata)
    # import random
    # random.shuffle(iqadata1)
        #     with open(json_) as f:
        #         iqadata = json.load(f)
            num=0
            prs, gts = [], []
            for i, llddata in enumerate(tqdm(iqadata, desc="Evaluating [{}]".format(json_.split("/")[-1]))):
                # inp = llddata['conversations'][0]['value']
                try:
                    filename = llddata["video_path"]
                except:
                    filename = llddata["img_path"]

                llddata["logits"] = defaultdict(float)
                                # inp = "The key frames of this video are:" + "\n" + DEFAULT_IMAGE_TOKEN + ". And the motion feature of the video is" + "\n" + DEFAULT_IMAGE_TOKEN + ". How would you rate the quality of this video?"
                                # # for name in os.listdir(image_path):
                # inp = DEFAULT_IMAGE_TOKEN+DEFAULT_IMAGE_TOKEN+"How would you rate the overall quality of this video?"
                inp = DEFAULT_IMAGE_TOKEN + DEFAULT_IMAGE_TOKEN
                # for name in os.listdir(image_path):
                    # if name.split('_')[0] in filename:
                try:
                    # for name in os.listdir(image_path):
                        # print(i)
                        image, frame_idx = load_video(os.path.join(image_path, filename), 1)
                        inp = inp
                        # image = None
                        cur_prompt = args.extra_prompt + inp
                        # print(cur_prompt)
                        # conv = conv_templates[args.conv_mode].copy()
                        # conv.append_message(conv.roles[0], cur_prompt)
                        # conv.append_message(conv.roles[1], "The quality of the video is")
                        # prompt = conv.get_prompt()

                        input_ids = preprocess_qwen([cur_prompt, {'from': 'gpt', 'value': ""}],
                                                    tokenizer,
                                                    has_image=True).cuda()
                        # print(input_ids)
                        image_tensor = image_processor.preprocess(image[:len(image) // 4 * 4], return_tensors='pt')[
                            'pixel_values']
                        image_tensor1 = \
                            image_processor.preprocess([image[frame_idx[i]] for i in range(len(frame_idx))],
                                                       return_tensors='pt')[
                                'pixel_values']
                        # for image_file in image:
                        #     image_tensor1 = transformations_test(image_file)
                        #     image_tensors.append(image_tensor1)
                        # image_tensors = torch.stack(image_tensors)
                        image_tensors = [[image_tensor[:image_tensor.shape[0] // 4 * 4].half().cuda()],
                                         [image_tensor1.half().cuda()]]

                        # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                        # keywords = [stop_str]
                        # stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
                        # TEXT = tokenizer.batch_decode(torch.argmax(model(input_ids,
                        #                                                  images=image_tensors)["logits"], dim=-1))
                        # print(TEXT)

                        # model = resnet50()
                        # input = torch.randn(1, 3, 224, 224)
                        # macs, params = profile(model, inputs=(input_ids, image_tensors,None,None,None,None,None,None,None,None,None,None,None,["video"],False,None))
                        # # macs = profile_macs(model, (input_ids, image_tensors,None,None,None,None,None,None,None,None,None,None,None,["video"],False,None))
                        # print(macs)

                        # 转换为 GFLOPs
                        # gflops = macs / 1e9
                        # print(f"Total GFLOPs: {gflops:.4f}")
                        # print(torch.argmax(output_logits,2).cpu().numpy()[0,-10:])
                        # print(tokenizer.batch_decode(torch.argmax(output_logits,2).cpu().numpy(), skip_special_tokens=False)[0])

                        output_logits = model(input_ids,
                                              images=image_tensors)["logits"]
                        # print(tokenizer.batch_decode(torch.argmax(output_logits, 2)))
                        # print(torch.argmax(output_logits,2))
                        output_logits=output_logits[:, -3]
                        skip_token_id = [198, 13, 151643, 151644, 151645]
                        # print(llddata["logits"])
                        llddata["logits"]["high"] += output_logits.mean(0)[1550].item()
                        llddata["logits"]["good"] += output_logits.mean(0)[1661].item()
                        llddata["logits"]["fair"] += output_logits.mean(0)[6624].item()
                        llddata["logits"]["poor"] += output_logits.mean(0)[7852].item()
                        llddata["logits"]["low"] += output_logits.mean(0)[3347].item()
                        # else:
                        #     llddata["logits"][tok] += output_logits.mean(0)[id_[0]].item()*output_logits.mean(0)[id_[1]].item()
                        llddata["score"] = wa5(llddata["logits"])
                        # print(llddata["score"])
                        prs.append(llddata["score"])
                        # gts.append(float(llddata["mos"]))
                        gts.append(float(llddata["mos"]))
                        # print(llddata)
                        json_ = json_.replace("combined/", "combined-")
                        mse = np.mean((np.array(gts) - np.array(prs)) ** 2)
                        # with open(f"results/{args.model_path}/{json_.split('/')[-1]}", "a") as wf:
                        #     json.dump(llddata, wf)

                        if i > 0 and i % 10 == 0:
                            print(spearmanr(prs, gts)[0], pearsonr(prs, gts)[0],mse)
                            with open(f"/tos-bjml-researcheval/jiaziheng/VQA++/Visual-Question-Answering-for-Video-Quality-Assessment/VQA_main/results/{args.model_path.split('/')[-1]}/{json_.split('/')[-1]}", "a") as wf:
                                                    json.dump([spearmanr(prs,gts)[0],pearsonr(prs,gts)[0]], wf)

                except:
                    for name in os.listdir(image_path):

                        if filename[:-4] in name:

                            num+=1
                            # if num==263:
                            #     continue
                            # print(num)
                            # print(os.path.join(image_path, name))
                            try:
                                image, frame_idx = load_video(os.path.join(image_path, name), 24)
                            except:
                                continue

                        # print(image_path)
                            inp = inp
                            # image = None
                            cur_prompt = args.extra_prompt + inp
                            # print(cur_prompt)
                            # conv = conv_templates[args.conv_mode].copy()
                            # conv.append_message(conv.roles[0], cur_prompt)
                            # conv.append_message(conv.roles[1], "The quality of the video is")
                            # prompt = conv.get_prompt()

                            input_ids = preprocess_qwen(
                                [cur_prompt, {'from': 'gpt', 'value': ""}], tokenizer,
                                has_image=True).cuda()
                            # input_ids=tokenizer_image_token( prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
                            #     0).cuda()
                            img_num = list(input_ids.squeeze()).count(IMAGE_TOKEN_INDEX)

                            image_tensors = []
                            # transformations_test = transforms.Compose(
                            #     [transforms.Resize([224, 224]), transforms.ToTensor(), \
                            #      transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])])
                            # for image_file in image:
                            # image = Image.open(os.path.join(args.image_folder, image_file))
                            image_tensor = image_processor.preprocess(image[:len(image) // 4 * 4], return_tensors='pt')[
                                'pixel_values']
                            image_tensor1 = \
                                image_processor.preprocess([image[frame_idx[i]] for i in range(len(frame_idx))],
                                                           return_tensors='pt')[
                                    'pixel_values']
                            # for image_file in image:
                            #     image_tensor1 = transformations_test(image_file)
                            #     image_tensors.append(image_tensor1)
                            # image_tensors = torch.stack(image_tensors)
                            image_tensors = [[image_tensor[:image_tensor.shape[0] // 4 * 4].half().cuda()],
                                             [image_tensor1.half().cuda()]]

                            # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                            # keywords = [stop_str]
                            # stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
                            # TEXT = tokenizer.batch_decode(torch.argmax(model(input_ids,
                            #                                                  images=image_tensors)["logits"], dim=-1))
                            # print(TEXT)


                            # output_logits = model(input_ids,
                            #                       images=image_tensors)["logits"][:, -3]
                            output_logits = model(input_ids,
                                                  images=image_tensors)["logits"]
                            # print(tokenizer.batch_decode(torch.argmax(output_logits, 2)))
                            # print(torch.argmax(output_logits,2)[:,-20:])
                            output_logits = output_logits[:, -3]
                            # macs = profile_macs(model, input_ids, images=image_tensors)
                            # print(macs)
                            #
                            # # 转换为 GFLOPs
                            # gflops = macs / 1e9
                            # print(f"Total GFLOPs: {gflops:.4f}")
                            # print(torch.argmax(output_logits,2).cpu().numpy()[0,-10:])
                            # print(tokenizer.batch_decode(torch.argmax(output_logits,2).cpu().numpy(), skip_special_tokens=False)[0])
                            skip_token_id = [198, 13, 151643, 151644, 151645]
                            llddata["logits"]["high"] += output_logits.mean(0)[1550].item()
                            llddata["logits"]["good"] += output_logits.mean(0)[1661].item()
                            llddata["logits"]["fair"] += output_logits.mean(0)[6624].item()
                            llddata["logits"]["poor"] += output_logits.mean(0)[7852].item()
                            llddata["logits"]["low"] += output_logits.mean(0)[3347].item()
                            # else:
                            #     llddata["logits"][tok] += output_logits.mean(0)[id_[0]].item()*output_logits.mean(0)[id_[1]].item()
                            llddata["score"] = wa5(llddata["logits"])
                            # print(llddata["mos"])
                            # print(llddata["score"])
                            # print(llddata)
                            prs.append(llddata["score"])
                            # gts.append(float(llddata["mos"]))
                            gts.append(float(llddata["mos"]))
                            # print(llddata)
                            json_ = json_.replace("combined/", "combined-")
                            # with open(f"results/{args.model_path}/{json_.split('/')[-1]}", "a") as wf:
                            #     json.dump(llddata, wf)

                            if i > 0 and i % 10 == 0:
                                print(spearmanr(prs, gts)[0], pearsonr(prs, gts)[0])
                            # except:
                            #     continue
                                with open(f"/tos-bjml-researcheval/jiaziheng/VQA++/Visual-Question-Answering-for-Video-Quality-Assessment/VQA_main/results/{args.model_path.split('/')[-1]}/{json_.split('/')[-1]}", "a") as wf:
                                                    json.dump([spearmanr(prs,gts)[0],pearsonr(prs,gts)[0]], wf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model-path", type=str, default="/fs-computility/mllm1/zhangzicheng/VQA++/LLaVA-NeXT-main7_supp/llava-ov-chat-qwen2-VQA++_rater_PLUS")
    parser.add_argument("--model-path", type=str, default="/fs-computility/ResearchEval/shared/zihengjia/llava-ov-chat-qwen2-VQA++_focal1")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--extra-prompt", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--test_size", type=int, default=10000000)
    args = parser.parse_args()

    eval_model(args)
