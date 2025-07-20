from transformers import AutoModelForCausalLM, AutoTokenizer
# from abc import ABC
from langchain.llms.base import LLM
from typing import List, Optional
import re
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_deepseek import ChatDeepSeek
device = "cuda"  # the device to load the model onto


# 基于BaseLLM编写自定义的模型接口
class Qwen(LLM):
    # 加载模型和分词器
    model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
        "../Qwen/Qwen3-0.6B",
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained("../Qwen/Qwen3-0.6B")

    def __init__(self, model_id: str = None):
        super().__init__()
        if model_id is not None:
            # 加载模型和分词器
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype="auto",
                device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        print('A6')

    @property
    def _llm_type(self) -> str:
        return "Qwen"

    # 编写模型的推理逻辑，传入prompt（也就是用户的问题），输出答案
    def _call(
         self,
         prompt: str,
         stop: Optional[List[str]] = None,
         run_manager: Optional[CallbackManagerForLLMRun] = None,
     ) -> str:
        # 检查是否为结构化提示词
        if prompt.strip().startswith("System:") and "Human:" in prompt:
            # 提取 System 和 Human 内容
            system_match = re.search(r"System:\s*(.*?)\s*Human:", prompt, re.DOTALL)
            user_match = re.search(r"Human:\s*(.*)", prompt, re.DOTALL)
            system_content = system_match.group(1).strip() if system_match else "You are a helpful assistant."
            user_content = user_match.group(1).strip() if user_match else ""
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
        else:
            # 普通提示词
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        text = self.tokenizer.apply_chat_template(
             messages,
             tokenize=False,
             add_generation_prompt=True
         )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(device)
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
            attention_mask=model_inputs["attention_mask"],
            pad_token_id=self.tokenizer.eos_token_id,  # 显式设置 pad_token_id
         )
        response = self.tokenizer.batch_decode(generated_ids,
                                               skip_special_tokens=True)[0]
        return response.split('\n')[-1]