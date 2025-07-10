from vllm import ModelRegistry
from llava.model.language_model.llava_qwen_vllm import LlavaQwenForCausalLM
ModelRegistry.register_model("LlavaQwenForCausalLM", LlavaQwenForCausalLM)
import runpy
runpy.run_module('vllm.entrypoints.openai.api_server', run_name='__main__')