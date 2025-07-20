from transformers import AutoModelForCausalLM, AutoTokenizer
from abc import ABC
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
device = "cuda" # the device to load the model onto

# 基于BaseLLM编写自定义的模型接口
class Qwen(LLM, ABC):
    # 加载模型和分词器
    model:AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
        "../Qwen/Qwen3-0.6B",
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer:AutoTokenizer = AutoTokenizer.from_pretrained("../Qwen/Qwen3-0.6B")

    def __init__(self):
        super().__init__()

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
             attention_mask=model_inputs["attention_mask"]
         )
         generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
            attention_mask=model_inputs["attention_mask"],
            pad_token_id=self.tokenizer.eos_token_id,  # 显式设置 pad_token_id
         )
         response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
         return response.split('\n')[-1]