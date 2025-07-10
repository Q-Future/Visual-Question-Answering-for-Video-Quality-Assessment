import argparse
import sys
sys.path.append('./')
import os
import json
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
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


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]
def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for width, height in possible_resolutions:
        # Calculate the downscaled size to keep the aspect ratio
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)

        # Calculate effective and wasted resolutions
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit
def resize_and_pad_image(image, target_resolution):
    """
    Resize and pad an image to a target resolution while maintaining aspect ratio.

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    # Determine which dimension (width or height) to fill
    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        # Width will be filled completely
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        # Height will be filled completely
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    # Create a new image with the target size and paste the resized image onto it
    new_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image


def divide_to_patches(image, patch_size):
    """
    Divides an image into patches of a specified size.

    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The size of each patch.

    Returns:
        list: A list of PIL.Image.Image objects representing the patches.
    """
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)

    return patches


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (tuple): The size of the input image in the format (width, height).
        grid_pinpoints (str): A string representation of a list of possible resolutions.
        patch_size (int): The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    if isinstance(grid_pinpoints, str) and "x" in grid_pinpoints:
        assert patch_size in [224, 336, 384, 448, 512], "patch_size should be in [224, 336, 384, 448, 512]"
        # Use regex to extract the range from the input string
        matches = re.findall(r"\((\d+)x(\d+)\)", grid_pinpoints)
        range_start = tuple(map(int, matches[0]))
        range_end = tuple(map(int, matches[-1]))
        # Generate a matrix of tuples from (range_start[0], range_start[1]) to (range_end[0], range_end[1])
        grid_pinpoints = [(i, j) for i in range(range_start[0], range_end[0] + 1) for j in range(range_start[1], range_end[1] + 1)]
        # Multiply all elements by patch_size
        grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    width, height = select_best_resolution(image_size, possible_resolutions)
    return width // patch_size, height // patch_size

def process_anyres_image(image, processor):
    # try:
    #     image = Image.open(image).convert("RGB")
    # except Exception as exn:
    #     # print(f"Failed to open image {image_file}. Exception:", exn)
    #     raise exn

    image_size = image.size
    image=image.resize((384,384))
    image_size=(384,384)
    """
    Process an image with variable resolutions.

    Args:
        image (PIL.Image.Image): The input image to be processed.
        processor: The image processor object.
        grid_pinpoints (str): A string representation of a list of possible resolutions.

    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """
    # Convert grid_pinpoints from string to list
    grid_pinpoints=[
    [
      384,
      384
    ],
    [
      384,
      768
    ],
    [
      384,
      1152
    ],
    [
      384,
      1536
    ],
    [
      384,
      1920
    ],
    [
      384,
      2304
    ],
    [
      768,
      384
    ],
    [
      768,
      768
    ],
    [
      768,
      1152
    ],
    [
      768,
      1536
    ],
    [
      768,
      1920
    ],
    [
      768,
      2304
    ],
    [
      1152,
      384
    ],
    [
      1152,
      768
    ],
    [
      1152,
      1152
    ],
    [
      1152,
      1536
    ],
    [
      1152,
      1920
    ],
    [
      1152,
      2304
    ],
    [
      1536,
      384
    ],
    [
      1536,
      768
    ],
    [
      1536,
      1152
    ],
    [
      1536,
      1536
    ],
    [
      1536,
      1920
    ],
    [
      1536,
      2304
    ],
    [
      1920,
      384
    ],
    [
      1920,
      768
    ],
    [
      1920,
      1152
    ],
    [
      1920,
      1536
    ],
    [
      1920,
      1920
    ],
    [
      1920,
      2304
    ],
    [
      2304,
      384
    ],
    [
      2304,
      768
    ],
    [
      2304,
      1152
    ],
    [
      2304,
      1536
    ],
    [
      2304,
      1920
    ],
    [
      2304,
      2304
    ]
  ]
    if isinstance(grid_pinpoints, str) and "x" in grid_pinpoints:
        try:
            patch_size = processor.size[0]
        except Exception as e:
            patch_size = processor.size["shortest_edge"]
        assert patch_size in [224, 336, 384, 448, 512], "patch_size should be in [224, 336, 384, 448, 512]"
        # Use regex to extract the range from the input string
        matches = re.findall(r"\((\d+)x(\d+)\)", grid_pinpoints)
        range_start = tuple(map(int, matches[0]))
        range_end = tuple(map(int, matches[-1]))
        # Generate a matrix of tuples from (range_start[0], range_start[1]) to (range_end[0], range_end[1])
        grid_pinpoints = [(i, j) for i in range(range_start[0], range_end[0] + 1) for j in range(range_start[1], range_end[1] + 1)]
        # Multiply all elements by patch_size
        grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]

    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    best_resolution = select_best_resolution(image.size, possible_resolutions)
    image_padded = resize_and_pad_image(image, best_resolution)

    patches = divide_to_patches(image_padded, processor.crop_size["height"])

    # FIXME: this seems to be a bug that it resizes instead of pad.
    # but to keep it consistent with previous, i will keep it as it is
    # TODO: uncomment below to ablate with the padding
    if isinstance(processor.size, dict):
        shortest_edge = processor.size["shortest_edge"]
    else:
        shortest_edge = min(processor.size)
    image_original_resize = image.resize((shortest_edge, shortest_edge))
    # image_padded_square = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
    # image_original_resize = image_padded_square.resize((processor.size['shortest_edge'], processor.size['shortest_edge']))

    image_patches = [image_original_resize] + patches
    image_patches = [processor.preprocess(image_patch, return_tensors="pt")["pixel_values"][0] for image_patch in image_patches]
    return torch.stack(image_patches, dim=0),image_size

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
def load_image(video_file, video_fps):
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
    avg_fps = round(vr.get_avg_fps() / video_fps)
    # total_frame_num=len(vr)//avg_fps*avg_fps
    frame_idx1.extend([i for i in range((ii+1)*round(vr.get_avg_fps()), len(vr), avg_fps)])
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
    # os.makedirs(f"./{args.model_path.split('/')[-1]}/", exist_ok=True)
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    model.half()
    # Data
    from datasets import load_dataset

    ds = load_dataset("q-future/Q-Bench2-HF",cache_dir="./")
    llvqa_data=[]
    # for file in ['/tos-bjml-researcheval/jiaziheng/qbench/llvisionqa_test.json']:
    check_questions=0
    correct=0
    for i, llddata in enumerate((tqdm(ds["dev"]))):


            image_file0= llddata['image1']
            image_file0=process_anyres_image(image_file0,image_processor)
            image_file1= llddata['image2']
            image_file1=process_anyres_image(image_file1,image_processor)

            image1 = image_file0
            image2 = image_file1
            image_tensors = [[[image1[0].repeat(4, 1, 1, 1).half()], [image1[0].half()]],[[image2[0].repeat(4, 1, 1, 1).half()], [image2[0].half()]]]
                    # image_tensors = [[[image_tensor[:image_tensor.shape[0] // 4 * 4].half().cuda()],
                    #                   [image_tensor[frame_idx].half().cuda()]]]


            message = llddata["question"] + "\n"
                # 设计选项
            for choice, ans in zip(["A.", "B.", "C.", "D."], [llddata["option0"],llddata["option1"],llddata["option2"],llddata["option3"]]):
                    message += f"{choice} {ans}\n"
            message = message + "Please answer the question in the following format: the uppercase letter of the correct answer option itself +'.'. Please do not add any other answers beyond this."

            prefix_text = "You will receive two images. Please analyze these images and answer the questions based on your observations."
                # prefix_text = "You will receive " + str(
                #     slice_len) + f" distinct frames that have been uniformly sampled in each second from a video sequence, arranged in the same temporal order as they appear in the video.  Please analyze these images and answer the questions based on your observations."
                # # print(message)
                # frames, frame_timestamps = load_video(video_file)
            prompt = prefix_text + '\n' + "<Video><VideoHere></Video>\n"+ '\n' + "<Video><VideoHere></Video>\n"+message
                # prompt = prefix_text + '\n' +  message
                # prompt = prefix_text + message + "<Video><VideoHere></Video>\n" + "<Video><VideoHere></Video>\n."
                # prompt = prefix_text + '\n' + 'The video frames:' + "<Video><VideoHere></Video>\n"  + message
            qs = prompt


            qs = qs.replace("<Video><VideoHere></Video>\n", DEFAULT_IMAGE_TOKEN  + '\n')

            cur_prompt = args.extra_prompt + qs
            print(cur_prompt)



            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], cur_prompt)
            conv.append_message(conv.roles[1], None)


            input_ids = preprocess_qwen([cur_prompt,{'from': 'gpt','value': None}], tokenizer, has_image=True).cuda()


            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]


            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensors,
                    image_sizes=[image1[1],image2[1]],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    modalities=["image"],
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    use_cache=True)


            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
            print('answer:'+outputs)

            check_questions += 1
            if llddata["correct_choice"]==outputs[0]:
                    correct += 1
            accuracy = correct / check_questions

            print(accuracy)
            llddata['answer'] = outputs
            del llddata["image1"]
            del llddata["image2"]

            llvqa_data.append(llddata)
            with open(f"/tos-bjml-researcheval/jiaziheng/VQA++/Visual-Question-Answering-for-Video-Quality-Assessment/VQA_benchmark_test/{args.model_path.split('/')[-1]}_" + f"pair_answer_image.json", "w") as f:
                json.dump(llvqa_data, f)
        # with open("llava-ov-7b_test_answer.json", "w") as f:
        #     json.dump(llvqa_data, f)

    # ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/tos-bjml-researcheval/jiaziheng/quality_foundation_model/train_foundation/model_output5/llava-ov-chat-qwen2-VQA++_foundation")
    # parser.add_argument("--model-path", type=str,
    #                     default="/DATA/DATA2/jzh/video_benchmark/LLaVA-NeXT-main/llava-ov-chat-qwen2_slowfast_base")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--extra-prompt", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--test_size", type=int, default=10000000)
    args = parser.parse_args()

    eval_model(args)
