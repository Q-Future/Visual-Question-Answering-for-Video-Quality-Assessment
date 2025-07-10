import argparse
import sys
import json


sys.path.append('./')
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
# os.environ['HF_ENDPOINT']= 'https://hf-mirror.com'
import torch
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from typing import Dict, Optional, Sequence, List
import transformers
import re

from PIL import Image
import math

dimension_dict={}

dimension_dict["sharpness"]=["What do you think about the sharpness of the video? Describe in detail.", "Can you describe the sharpness of the video during its playback?",  "How would you assess the sharpness of the video?", "How would you characterize the sharpness quality in the video from start to finish?", "How do you rate the sharpness of the video overall?"]
dimension_dict["light"]=["How would you evaluate the light condition and distribution of the video? Describe in detail.", "Can you describe the video’s overall exposure condition throughout its playback? Describe in detail.", "What is your evaluation of the overall light condition of the video?","What is your feeling of the light throughout the video?", "What is your description on the video’s light conditions?"]
dimension_dict["compression_artifact"]=["How would you assess the presence of compression artifacts in the video? Describe in detail.", "To what extent are compression artifacts visible in this video?", "How is the degree of the compression artifacts throughout the video?","Can you comment on the condition of the compression artifacts in the video?","What do you think about the severity of compression artifacts in the video?"]
dimension_dict["flickers"]=["How noticeable is the flicker throughout the video? Describe in detail.", "How would you describe the level of flicker during playback? Please describe in detail.", "Can you comment on the condition of the flicker in the video?", "How is your feeling on the flicker condition?", "What is your opinion on the condition of the camera shake in this video?"]
dimension_dict["fluency"]=["What is your evaluation of the overall fluency of the video? Please describe in detail.", "How would you rate the smoothness of the video throughout its playback?", "Can you describe the fluency of the video from start to finish?","How does the video condition in terms of smoothness and fluency?", "What do you feel about the video’s playback fluency?"]
dimension_dict["color"]=["How would you evaluate the color of this video? Describe in detail.","How is the color condition of this video?", "How is the color tone of this video?","How is the colorfulness of this video?","Describe the overall color condition."]
dimension_dict["motion blur"]=["How is the degree of the motion blur of the video?", "Are there any noticeable motion blur in this video?", "How would you evaluate the motion blur condition of this video? Describe in detail.", "What is the motion blur condition of this video?", "Describe the motion blur condition in this video."]
dimension_dict["noise"]=["What is the noise level in this video?", "How is the noise condition of this video? Describe in detail.", "Are there any noticeable noise in this video?", "What is the noise condition in the video?", "Describe the noise condition in this video."]


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

# def load_video(video_file,frames_limit):
#     from decord import VideoReader
#     vr = VideoReader(video_file)
#
#     # Get video frame rate获得帧率
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
    frame_num=len(vr)
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
    ii=-1
    for ii in range(len(vr)//round(vr.get_avg_fps())):
        total_frame_num = round(vr.get_avg_fps())
        avg_fps = round(vr.get_avg_fps() / video_fps)
        # total_frame_num=len(vr)//avg_fps*avg_fps
        frame_idx1.extend([i for i in range(ii*round(vr.get_avg_fps()), (ii+1)*round(vr.get_avg_fps()), avg_fps)])
    total_frame_num = len(vr)-(len(vr)//round(vr.get_avg_fps())*round(vr.get_avg_fps()))
    # avg_fps = round((len(vr)-ii*vr.get_avg_fps()) / video_fps)
    # # total_frame_num=len(vr)//avg_fps*avg_fps
    # frame_idx1.extend([i for i in range((ii+1)*round(vr.get_avg_fps()), len(vr), avg_fps)])
    # if len(frame_idx) > 200:
    #             uniform_sampled_frames = np.linspace(0, total_frame_num - 1, 100, dtype=int)
    #             frame_idx = uniform_sampled_frames.tolist()

    return [Image.fromarray(frames[i]) for i in range(len((vr)))],frame_idx1,frame_num//4*4

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
                _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence).input_ids + [im_end] + nl_tokens
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
    os.makedirs(f"./{args.model_path.split('/')[-1]}/", exist_ok=True)
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    for param in model.parameters():
        print(param)
    video_overall_annotation=[]
    # Data
    # base_video_path = "/DATA/DATA2/jzh/video_benchmark/LLaVA-NeXT-main_benchmark/video/"
    # image_files,frame_idx,frame_num = load_video(base_video_path+"LSVQ_fair_3001-3500/videos\yfcc-batch23_18710.mp4",1)
    base_video_path = "/fs-computility/mllm1/zhangzicheng/VQA++/VIDEOS/"

    for i, video in enumerate(os.listdir(base_video_path)[10:]):
            video_information = {}
            video_information["video_path"] = video
            image_files, frame_idx, frame_num = load_video(base_video_path + video,1)
            slice_len = len(frame_idx)
            if slice_len < 3:
                continue
            print(video)
            for i in range(slice_len):
                video_information[f"{i}s"] = {}
                image_tensor = image_processor.preprocess(image_files, return_tensors='pt')['pixel_values']
                image_tensors = [[[image_tensor[:image_tensor.shape[0]][image_tensor.shape[0]//slice_len*i//4*4:min(image_tensor.shape[0]//slice_len*(i+1)//4*4,image_tensor.shape[0]//4*4)].half().cuda()],
                                  [image_tensor[frame_idx[i]].unsqueeze(0).half().cuda()]]]
                frame_num=image_tensors[0][0][0].shape[0]
                for key in dimension_dict.keys():
                    video_information[f"{i}s"][key] = []
                    for message in dimension_dict[key]:
                        print(message)
                        message = message
                        prefix_text = "You will receive " + str(1) + f" keyframes that have been sampled from one video-clip with 1s long. In addition, you will also obtain motion features extracted from all {frame_num} frames of the the video clip. Please analyze the keyframes and answer the questions based on your observations."
                        prompt = prefix_text + '\n' + 'The video frames:' + "<Video><VideoHere></Video>\n" + 'The video motion features:' + "<Video><VideoHere></Video>\n" + message
                        qs = prompt
                        qs = qs.replace("<Video><VideoHere></Video>\n", DEFAULT_IMAGE_TOKEN + '\n')
                        cur_prompt = args.extra_prompt + qs
                        conv = conv_templates[args.conv_mode].copy()
                        conv.append_message(conv.roles[0], cur_prompt)
                        conv.append_message(conv.roles[1], None)
                        prompt = conv.get_prompt()
                        input_ids = preprocess_qwen([cur_prompt, {'from': 'gpt', 'value': None}], tokenizer,
                                                    has_image=True).cuda()
                        img_num = list(input_ids.squeeze()).count(IMAGE_TOKEN_INDEX)
                        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                        keywords = [stop_str]
                        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
                        with torch.inference_mode():
                            output_ids = model.generate(
                                input_ids,
                                images=image_tensors,
                                do_sample=True if args.temperature > 0 else False,
                                temperature=args.temperature,
                                top_p=args.top_p,
                                num_beams=args.num_beams,
                                max_new_tokens=1024,
                                use_cache=True)
                        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                        outputs = outputs.strip()
                        if outputs.endswith(stop_str):
                            outputs = outputs[:-len(stop_str)]
                        outputs = outputs.strip()
                        print('answer:' + outputs)
                        video_information[f"{i}s"][key].append(outputs)
            video_overall_annotation.append(video_information)
            with open('./video_overall_annotation1' + ".json", "w") as f:
                json.dump(video_overall_annotation, f, indent=4)

        # with open(f"./{args.model_path.split('/')[-1]}/" + "answer.json", "w") as f:
        #     json.dump(llvqa_data, f)
        # with open(f"./{args.model_path.split('/')[-1]}/" + "result.json", "w") as f:
        #     json.dump({'num_of_questions': check_questions, 'correct_num': correct, 'accuracy': accuration}, f)

        # ans_id = shortuuid.uuid()
        # ans_file.write(json.dumps({
        #                            "dataset": dataset_name,
        #                            "sample_id": idx,
        #                            "prompt": cur_prompt,
        #                            "pred_response": outputs,
        #                            "gt_response": gt,
        #                            "shortuuid": ans_id,
        #                            "model_id": model_name,
        #                            "question_type": question_type,
        #                            }) + "\n")
        # ans_file.flush()
        #
        # if len(line["conversations"]) > 2:
        #
        #     for i in range(2, len(line["conversations"]), 2):
        #         input_ids = torch.cat((input_ids, output_ids), dim=1)
        #
        #         gt = line["conversations"][i + 1]["value"]
        #         qs = line["conversations"][i]["value"]
        #         cur_prompt = args.extra_prompt + qs
        #
        #         args.conv_mode = "qwen_1_5"
        #
        #         conv = conv_templates[args.conv_mode].copy()
        #         conv.append_message(conv.roles[0], qs)
        #         conv.append_message(conv.roles[1], None)
        #         prompt = conv.get_prompt()
        #
        #         input_ids_new = preprocess_qwen([line["conversations"][i],{'from': 'gpt','value': None}], tokenizer, has_image=True).cuda()
        #         input_ids = torch.cat((input_ids, input_ids_new), dim=1)
        #         img_num = list(input_ids_new.squeeze()).count(IMAGE_TOKEN_INDEX)
        #
        #         stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        #         keywords = [stop_str]
        #         stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        #
        #         with torch.inference_mode():
        #             output_ids = model.generate(
        #                 input_ids,
        #                 images=image_tensors,
        #                 do_sample=True if args.temperature > 0 else False,
        #                 temperature=args.temperature,
        #                 top_p=args.top_p,
        #                 num_beams=args.num_beams,
        #                 # no_repeat_ngram_size=3,
        #                 max_new_tokens=1024,
        #                 use_cache=True)
        #
        #         outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        #         outputs = outputs.strip()
        #         if outputs.endswith(stop_str):
        #             outputs = outputs[:-len(stop_str)]
        #         outputs = outputs.strip()
        #
        #         ans_id = shortuuid.uuid()
        #         ans_file.write(json.dumps({
        #                                 "dataset": dataset_name,
        #                                 "sample_id": idx,
        #                                 "prompt": cur_prompt,
        #                                 "pred_response": outputs,
        #                                 "gt_response": gt,
        #                                 "shortuuid": ans_id,
        #                                 "model_id": model_name,
        #                                 "question_type": question_type,
        #                                 }) + "\n")
        #         ans_file.flush()


    # ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model-path", type=str, default="/DATA/DATA2/jzh/video_benchmark/LLaVA-NeXT-main/llava-ov-chat-qwen2_slowfast_base")
    parser.add_argument("--model-path", type=str,default="/fs-computility/mllm1/zhangzicheng/VQA++/llava-ov-chat-qwen2-stage123_final")
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
