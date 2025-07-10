# from .demo_modelpart import InferenceDemo
import gradio as gr
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['HF_ENDPOINT']= 'https://hf-mirror.com'
import sys
sys.path.append('./')
# os.environ["http_proxy"] = 'http://202.120.62.181:24097'
# os.environ["https_proxy"]='http://202.120.62.181:24097'
# import time
import cv2


# import copy
import torch

import spaces
import numpy as np

from llava import conversation as conversation_lib
from llava.constants import DEFAULT_IMAGE_TOKEN


from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer

import gradio as gr
import gradio_client
import subprocess
import sys

def install_gradio_4_35_0():
    current_version = gr.__version__
    if current_version != "4.35.0":
        print(f"Current Gradio version: {current_version}")
        print("Installing Gradio 4.35.0...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gradio==4.35.0", "--force-reinstall"])
        print("Gradio 4.35.0 installed successfully.")
    else:
        print("Gradio 4.35.0 is already installed.")

# Call the function to install Gradio 4.35.0 if needed
# install_gradio_4_35_0()

import gradio as gr
import gradio_client
print(f"Gradio version: {gr.__version__}")
print(f"Gradio-client version: {gradio_client.__version__}")

class InferenceDemo(object):
    def __init__(
        self, args, model_path, tokenizer, model, image_processor, context_len
    ) -> None:
        disable_torch_init()

        self.tokenizer, self.model, self.image_processor, self.context_len = (
            tokenizer,
            model,
            image_processor,
            context_len,
        )

        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        elif "qwen" in model_name.lower():
            conv_mode = "qwen_1_5"
        else:
            conv_mode = "llava_v0"

        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print(
                "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                    conv_mode, args.conv_mode, args.conv_mode
                )
            )
        else:
            args.conv_mode = conv_mode
        self.conv_mode = conv_mode
        self.conversation = conv_templates[args.conv_mode].copy()
        self.num_frames = args.num_frames


def is_valid_video_filename(name):
    video_extensions = ["avi", "mp4", "mov", "mkv", "flv", "wmv", "mjpeg"]

    ext = name.split(".")[-1].lower()

    if ext in video_extensions:
        return True
    else:
        return False


def sample_frames(video_file, num_frames):
    video = cv2.VideoCapture(video_file)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = total_frames // num_frames
    frames = []
    for i in range(total_frames):
        ret, frame = video.read()
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not ret:
            continue
        if i % interval == 0:
            frames.append(pil_img)
    video.release()
    return frames


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            print("failed to load the image")
    else:
        print("Load image from local file")
        print(image_file)
        image = Image.open(image_file).convert("RGB")

    return image
def load_video(video_file, video_fps):
    from decord import VideoReader,cpu
    vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
    frame_num=len(vr)
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
    frame_idx1.extend([i for i in range((ii+1)*round(vr.get_avg_fps()), len(vr), avg_fps)])
    return [Image.fromarray(frames[i]) for i in range(len((vr)))],frame_idx1,frame_num//4*4

def clear_history(history):

    our_chatbot.conversation = conv_templates[our_chatbot.conv_mode].copy()

    return None


def clear_response(history):
    for index_conv in range(1, len(history)):
        # loop until get a text response from our model.
        conv = history[-index_conv]
        if not (conv[0] is None):
            break
    question = history[-index_conv][0]
    history = history[:-index_conv]
    return history, question


# def print_like_dislike(x: gr.LikeData):
#     print(x.index, x.value, x.liked)


def add_message(history, message):
    # history=[]
    global our_chatbot
    if len(history) == 0:
        our_chatbot = InferenceDemo(
            args, model_path, tokenizer, model, image_processor, context_len
        )

    for x in message["files"]:
        history.append(((x,), None))
    if message["text"] is not None:
        history.append((message["text"], None))
    return history, gr.MultimodalTextbox(value=None, interactive=False)


@spaces.GPU
def bot(history):
    text = history[-1][0]
    images_this_term = []
    num_new_images = 0
    for i, message in enumerate(history[:-1]):
        if type(message[0]) is tuple:
            images_this_term.append(message[0][0])
            if is_valid_video_filename(message[0][0]):
                num_new_images += our_chatbot.num_frames
            else:
                num_new_images += 1
        else:
            num_new_images = 0
    assert len(images_this_term) > 0, "must have an image"
    for f in images_this_term:
        if ".mp4" in f:
            image_files, frame_idx, frame_num=load_video(f,1)
    # image_tensor = [
    #     our_chatbot.image_processor.preprocess(f, return_tensors="pt")["pixel_values"][
    #         0
    #     ]
    #     .half()
    #     .to(our_chatbot.model.device)
    #     for f in image_list
    # ]

    # image_tensor = torch.stack(image_tensor)
            image_token = DEFAULT_IMAGE_TOKEN
            # if our_chatbot.model.config.mm_use_im_start_end:
            #     inp = DEFAULT_IM_START_TOKEN + image_token + DEFAULT_IM_END_TOKEN + "\n" + inp
            # else:
            # inp = text
            # inp = image_token + "\n" + inp
            # image_files, frame_idx, frame_num = load_video(base_video_path + "3206.mp4",
            #                                                1)
            slice_len = len(frame_idx)
            image_tensor = our_chatbot.image_processor.preprocess(image_files, return_tensors='pt')['pixel_values']
            # for image_file in image:
            #     image_tensor1 = transformations_test(image_file)
            #     image_tensors.append(image_tensor1)
            # image_tensors = torch.stack(image_tensors)
            image_tensors = [
                [[image_tensor[:image_tensor.shape[0] // 4 * 4].half().cuda()], [image_tensor[frame_idx].half().cuda()]]]

    # message = "Is the man singing in the video fluent?"
    # # message ="Does motion blur happen in the video?"
    # # message ="When is the motion blur the most severe?"
    # message = message

            prefix_text = "You will receive " + str(
                    slice_len) + f" distinct frames that have been by uniformly sampling {1} frame per second from the video, arranged in the same temporal order as they appear in the video. In addition, you will also obtain motion features extracted from all {frame_num} frames of the the video."
            print(len(history))
            prompt = prefix_text + '\n' + 'The video frames:' + "<Video><VideoHere></Video>\n" + 'The video motion features:' + "<Video><VideoHere></Video>\n" + text
            qs = prompt
            qs = qs.replace("<Video><VideoHere></Video>\n", image_token + '\n')





    our_chatbot.conversation.append_message(our_chatbot.conversation.roles[0], qs)
    # print(our_chatbot.conversation.roles[0])
    # image = None
    our_chatbot.conversation.append_message(our_chatbot.conversation.roles[1], None)
    prompt = our_chatbot.conversation.get_prompt()

    print(prompt)
    input_ids = (
        tokenizer_image_token(
            prompt, our_chatbot.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        .unsqueeze(0)
        .to(our_chatbot.model.device)
    )
    stop_str = (
        our_chatbot.conversation.sep
        if our_chatbot.conversation.sep_style != SeparatorStyle.TWO
        else our_chatbot.conversation.sep2
    )
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(
        keywords, our_chatbot.tokenizer, input_ids
    )
    streamer = TextStreamer(
        our_chatbot.tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    # print(our_chatbot.model.device)
    # print(input_ids.device)
    # print(image_tensor.device)
    # import pdb;pdb.set_trace()
    if text=='How would you rate the quality of the video?':
        def wa5(logits):
            import numpy as np
            logprobs = np.array([logits["high"], logits["good"], logits["fair"], logits["poor"], logits["low"]])
            probs = np.exp(logprobs) / np.sum(np.exp(logprobs))
            return np.inner(probs, np.array([1, 0.75, 0.5, 0.25, 0.]))
        with torch.inference_mode():
            output_logits = model(input_ids,
                                  images=image_tensors)["logits"][:, -3]
            llddata={}
            llddata["logits"]={}
            # print(torch.argmax(output_logits,2).cpu().numpy()[0,-10:])
            # print(tokenizer.batch_decode(torch.argmax(output_logits,2).cpu().numpy(), skip_special_tokens=False)[0])
            skip_token_id = [198, 13, 151643, 151644, 151645]
            llddata["logits"]["high"] = output_logits.mean(0)[1550].item()
            llddata["logits"]["good"] = output_logits.mean(0)[1661].item()
            llddata["logits"]["fair"] = output_logits.mean(0)[6624].item()
            llddata["logits"]["poor"] = output_logits.mean(0)[7852].item()
            llddata["logits"]["low"] = output_logits.mean(0)[3347].item()
            # else:
            #     llddata["logits"][tok] += output_logits.mean(0)[id_[0]].item()*output_logits.mean(0)[id_[1]].item()
            llddata["score"] = wa5(llddata["logits"])
            outputs=str(llddata["score"])
    else:
        with torch.inference_mode():
            output_ids = our_chatbot.model.generate(
                input_ids,
                images=image_tensors,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

            outputs = our_chatbot.tokenizer.decode(output_ids[0]).strip()
            if outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)]
    our_chatbot.conversation.messages[-1][-1] = outputs

    history[-1] = [text, outputs]
    # our_chatbot.conversation = conv_templates[our_chatbot.conv_mode].copy()
    return history


txt = gr.Textbox(
    scale=4,
    show_label=False,
    placeholder="Enter text and press enter.",
    container=False,
)

with gr.Blocks(
    css=".message-wrap.svelte-1lcyrx4>div.svelte-1lcyrx4  img {min-width: 20px}",
) as demo:
    gr.Markdown("<h2 class='title'>VQA²-Assistant (7B)</h2>")
    gr.Markdown("I can assist you in evaluating theseverity of a specific distortion in avideo, identifying the location andtime where the distortion occurs.and providing effective post-processing or reshootingrecommendations to mitigate thedistortion. Feel free to ask me!")

    # Informations
    # title_markdown = """
    #     # LLaVA-NeXT Interleave
    #     [[Blog]](https://llava-vl.github.io/blog/2024-06-16-llava-next-interleave/)  [[Code]](https://github.com/LLaVA-VL/LLaVA-NeXT) [[Model]](https://huggingface.co/lmms-lab/llava-next-interleave-7b)
    #     Note: The internleave checkpoint is updated (Date: Jul. 24, 2024), the wrong checkpiont is used before.
    # """
    # tos_markdown = """
    # ### TODO!. Terms of use
    # By using this service, users are required to agree to the following terms:
    # The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. The service may collect user dialogue data for future research.
    # Please click the "Flag" button if you get any inappropriate answer! We will collect those to keep improving our moderator.
    # For an optimal experience, please use desktop computers for this demo, as mobile devices may compromise its quality.
    # """
    # learn_more_markdown = """
    # ### TODO!. License
    # The service is a research preview intended for non-commercial use only, subject to the model [License](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) of LLaMA, [Terms of Use](https://openai.com/policies/terms-of-use) of the data generated by OpenAI, and [Privacy Practices](https://chrome.google.com/webstore/detail/sharegpt-share-your-chatg/daiacboceoaocpibfodeljbdfacokfjb) of ShareGPT. Please contact us if you find any potential violation.
    # """
    models = [
        "VQA²-Assistant",
    ]
    cur_dir = os.path.dirname(os.path.abspath(__file__))



    # gr.Markdown(title_markdown)
    with gr.Column():

        with gr.Row():
            chatbot = gr.Chatbot([],elem_id="chatbot",value=[("System", "Welcome to VQA²-Assistant chatbot!")], bubble_full_width=False,height=1200)
        # with gr.Blocks(css=".title { font-size: 24px; font-weight: bold; color: #333; }") as demo:
        #     gr.Markdown("<h2 class='title'>VQA²-Assistant</h2>")
        #     gr.Markdown("This chatbot assists with Video Quality Assessment (VQA) and related queries.")
        #     chatbot.render()
        with gr.Row():
            upvote_btn = gr.Button(value="👍  Upvote", interactive=True)
            downvote_btn = gr.Button(value="👎  Downvote", interactive=True)
            flag_btn = gr.Button(value="⚠️  Flag", interactive=True)
            # stop_btn = gr.Button(value="⏹️  Stop Generation", interactive=True)
            # regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=True)
            clear_btn = gr.Button(value="🗑️  Clear history", interactive=True)

        chat_input = gr.MultimodalTextbox(
            interactive=True,
            file_types=["image", "video"],
            placeholder="Enter message or upload file...",
            show_label=False,
        )

        print(cur_dir)
        gr.Examples(
            examples=[
                [
                    {
                        "text": "Which image shows a different mood of character from the others?",
                        "files": [f"{cur_dir}/examples/examples_image12.jpg", f"{cur_dir}/examples/examples_image13.jpg", f"{cur_dir}/examples/examples_image14.jpg"]
                    },
                    {
                        "text": "Please pay attention to the movement of the object from the first image to the second image, then write a HTML code to show this movement.",
                        "files": [
                            f"{cur_dir}/examples/code1.jpeg",
                            f"{cur_dir}/examples/code2.jpeg",
                        ],
                    }
                ],
                [
                    {
                        "files": [
                            f"{cur_dir}/examples/shub.jpg",
                            f"{cur_dir}/examples/shuc.jpg",
                            f"{cur_dir}/examples/shud.jpg",
                        ],
                        "text": "what is fun about the images?",
                    }
                ],
                [
                    {
                        "files": [
                            f"{cur_dir}/examples/iphone-15-price-1024x576.jpg",
                            f"{cur_dir}/examples/dynamic-island-1024x576.jpg",
                            f"{cur_dir}/examples/iphone-15-colors-1024x576.jpg",
                            f"{cur_dir}/examples/Iphone-15-Usb-c-charger-1024x576.jpg",
                            f"{cur_dir}/examples/A-17-processors-1024x576.jpg",
                        ],
                        "text": "The images are the PPT of iPhone 15 review. can you summarize the main information?",
                    }
                ],
                [
                    {
                        "files": [
                            f"{cur_dir}/examples/fangao3.jpeg",
                            f"{cur_dir}/examples/fangao2.jpeg",
                            f"{cur_dir}/examples/fangao1.jpeg",
                        ],
                        "text": "Do you kown who draw these paintings?",
                    }
                ],
                [
                    {
                        "files": [
                            f"{cur_dir}/examples/oprah-winfrey-resume.png",
                            f"{cur_dir}/examples/steve-jobs-resume.jpg",
                        ],
                        "text": "Hi, there are two candidates, can you provide a brief description for each of them for me?",
                    }
                ],
                [
                    {
                        "files": [
                            f"{cur_dir}/examples/original_bench.jpeg",
                            f"{cur_dir}/examples/changed_bench.jpeg",
                        ],
                        "text": "How to edit image1 to make it look like image2?",
                    }
                ],
                [
                    {
                        "files": [
                            f"{cur_dir}/examples/twitter2.jpeg",
                            f"{cur_dir}/examples/twitter3.jpeg",
                            f"{cur_dir}/examples/twitter4.jpeg",
                        ],
                        "text": "Please write a twitter blog post with the images.",
                    }
                ]

            ],
            inputs=[chat_input],
            label="Compare images: "
        )

    chat_msg = chat_input.submit(
        add_message, [chatbot, chat_input], [chatbot, chat_input]
    )
    bot_msg = chat_msg.then(bot, chatbot, chatbot, api_name="bot_response")
    bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

    # chatbot.like(print_like_dislike, None, None)
    clear_btn.click(
        fn=clear_history, inputs=[chatbot], outputs=[chatbot], api_name="clear_all"
    )


demo.queue()

if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--server_name", default="0.0.0.0", type=str)
    argparser.add_argument("--port", default=8081, type=int)
    argparser.add_argument(
        "--model_path", default="/tos-bjml-researcheval/jiaziheng/quality_foundation_model/train_foundation/model_output/llava-ov-chat-qwen2-VQA++_foundation", type=str
    )
    # argparser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    argparser.add_argument("--model-base", type=str, default=None)
    argparser.add_argument("--num-gpus", type=int, default=1)
    argparser.add_argument("--conv-mode", type=str, default=None)
    argparser.add_argument("--temperature", type=float, default=0.2)
    argparser.add_argument("--max-new-tokens", type=int, default=1024)
    argparser.add_argument("--num_frames", type=int, default=16)
    argparser.add_argument("--load-8bit", action="store_true")
    argparser.add_argument("--load-4bit", action="store_true")
    argparser.add_argument("--debug", action="store_true")

    args = argparser.parse_args()

    model_path = args.model_path
    filt_invalid = "cut"
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit)
    model=model.to(torch.device('cuda'))
    model.half()
    our_chatbot = None
    demo.launch(server_name=args.server_name,server_port=args.port,share=True)
