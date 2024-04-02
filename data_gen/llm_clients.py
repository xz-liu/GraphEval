from args import args
from tqdm import tqdm
from typing import List, Optional
import openai
import signal
import torch


def add_system_prompt(question: str, model_type=None) -> str:
    if model_type == 'llama':

        return (
            # '<s>[INST] <<SYS>>\n'
                'Below is an instruction that describes a task, '
                'paired with an input that provides further context. '
                'Write a response that appropriately completes the request.\n\n'
                # '<</SYS>>\n\n'
                + question
                +'\n\n## Response:\n\n'
            # + '[/INST]'
        )
    elif model_type == 'gemma':
        return (
                '<start_of_turn>user'
                'Below is an instruction that describes a task, '
                'paired with an input that provides further context. '
                'Write a response that appropriately completes the request.\n\n'
                + question
                + '<end_of_turn>'
                + '<start_of_turn>model'
                +  '\n\n The answer is "'
        )
    elif model_type=='chatgpt':
        return (
            'Below is an instruction that describes a task, '
            'paired with an input that provides further context. '
            'Write a response that appropriately completes the request.\n\n'
            + question
            + '\n\n## Response:\n\n'
        )
    else:
        raise ValueError(f"model_type {model_type} not implemented")



class ChatGPT:
    def __init__(self, openai_key, model='gpt-3.5-turbo'):
        super(ChatGPT, self).__init__()
        openai.api_key = openai_key
        self.model = model

    def call_openai(self, prompt):
        completion = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }

            ]
        )
        return completion['choices'][0]['message']['content']

    def call_openai_with_timeout(self, prompt, timeout_duration=10):
        class TimeoutException(Exception):
            pass

        def handler(signum, frame):
            raise TimeoutException()

        # Set the timeout handler
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(timeout_duration)

        try:
            response = self.call_openai(prompt)
        except TimeoutException:
            response = None
        finally:
            # Cancel the alarm
            signal.alarm(0)

        return response

class AskModel:
    def __init__(self, size='7b', model_name=None, token=None, model_type=None):
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, LlamaTokenizer, LlamaForCausalLM, \
            LlamaConfig

        # from accelerate import Accelerator
        # accelerator = Accelerator()
        self.model_name = model_name
        if model_type=='chatgpt':
            if model_name=='chatgpt':
                model_name='gpt-3.5-turbo'
            self.chatgpt=ChatGPT(openai_key='openai_key', model=model_name)
            self.model_type=model_type
            return

        import transformers
        if model_name is None:
            model_name = f"meta-llama/Llama-2-{size}-chat-hf"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.config = AutoConfig.from_pretrained(model_name, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, config=self.config,
                                                          token=token, device_map='auto', use_flash_attention_2= not args.no_flash_attention,torch_dtype=torch.float16)
        self.model_name = model_name
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            # device='cuda',
            device_map='auto',

        )
        self.model_type = model_type

    def call_chatgpt(self,questions:List[str], default_respond='I don\'t know'):
        responses:List[str] = [None] * len(questions)
        def call_openai(prompts, chatgpt):
            for i, prompt in enumerate(prompts):
                yield chatgpt.call_openai_with_timeout(prompt)
        for i, resp in tqdm(enumerate(call_openai(questions, self.chatgpt))):
            if resp is None:
                resp= default_respond
            responses[i] = resp
        for i in range(len(questions)):
            print("----------")
            print("Question: ", questions[i])
            print("Answer: ", responses[i])
            print("----------")
        return responses
    @torch.no_grad()
    def __call__(self, questions: List[str], default_respond='exception E') -> List[str]:
        if self.model_type=='chatgpt':
            return self.call_chatgpt(questions,default_respond)
        responses = [None] * len(questions)

        def data_generator():
            for question in questions:
                yield add_system_prompt(question, self.model_type)

        llama_args = dict(
            top_k=10,  # default=10
            top_p=0.9,  # default=0.9
            temperature=0.6,  # default=0.
            batch_size=len(questions),
            do_sample=True,
        )
        if 'llama-2-7b' not in self.model_name:
            llama_args = {}
        print("llama_args: ", llama_args)
        pipeline = self.pipeline
        for i, resp in tqdm(enumerate(pipeline(data_generator(),
                                               num_return_sequences=1,
                                               eos_token_id=self.tokenizer.eos_token_id,
                                               max_new_tokens=4096,  # max lenght of output, default=4096
                                               return_full_text=False,  # to not repeat the question, set to False,
                                               **llama_args,
                                               ))):
            responses[i] = resp[0]['generated_text']
        for i in range(len(questions)):
            print("----------")
            print("Question: ", questions[i])
            print("Answer: ", responses[i])
            print("----------")
        return responses
    