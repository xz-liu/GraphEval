from typing import List, Dict
import torch
from torch import nn
import torch.utils.data

from transformers import AutoModel, AutoTokenizer, AutoConfig, LlamaTokenizer, LlamaModel, AutoModel, \
    LlamaForCausalLM, LlamaConfig, AutoModelForCausalLM
from trainer.llm_state_loader import *
from peft import PeftModel
from args import *
from utils import SharedStringList
import time

import ctypes

import torch.multiprocessing as mp


class EvaluationModel(torch.nn.Module):
    def __init__(self, peft_model, classifier):
        super(EvaluationModel, self).__init__()
        self.peft_model = peft_model
        self.classifier = classifier

    def forward(self, input_ids, attention_mask, true_false):
        peft_output = self.peft_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        state = peft_output.hidden_states[-1][:, -1, :]
        result = self.classifier(state, true_false)
        return result

    @staticmethod
    def from_path_dict(path_dict: Dict):
        '''
        :param path_dict:
        dict(
                model_name=model_name,
                model_type=model_type,
                peft_path=trained_prompt_encoder_path(model_name, test_case_name, which=kg.which),
                classifier_path=trained_clf_path(model_name, test_case_name, which=kg.which),
                model_args=model_args
            )
        :return:
        '''
        print('Init model', path_dict['model_name'], 'with args', path_dict['model_args'], ',use peft',
              not args.no_peft, '\nclassifier', path_dict['classifier_path'], '\npeft', path_dict['peft_path'])
        model = AutoModelForCausalLM.from_pretrained(small_lm_of(path_dict['model_name']), **path_dict['model_args'])
        if not args.no_peft:
            peft_model = PeftModel.from_pretrained(model, path_dict['peft_path'])
        else:
            peft_model = model
        classifier = load_from(path_dict['classifier_path'])
        return EvaluationModel(peft_model, classifier)


class EvaluationResultUnit:
    def __init__(self, h, r, t, original_h, original_r, original_t, prompt, corrupted, logits=None, prediction=None,
                 inference_time=0):
        self.h = int(h)
        self.r = int(r)
        self.t = int(t)
        self.original_h = int(original_h)
        self.original_r = int(original_r)
        self.original_t = int(original_t)
        self.prompt = prompt
        self.corrupted = bool(corrupted)
        if logits is not None:
            self.logits = torch.tensor(logits).detach().cpu()
        else:
            self.logits = None
        self.prediction = int(prediction) if prediction is not None else None
        self.inference_time = inference_time

    def triple(self) -> Tuple[int, int, int]:
        return self.h, self.r, self.t

    def original_triple(self) -> Tuple[int, int, int]:
        return self.original_h, self.original_r, self.original_t

    @staticmethod
    def from_dict(d: Dict):
        return EvaluationResultUnit(d.get('h'), d.get('r'), d.get('t'), d.get('original_h'), d.get('original_r'),
                                    d.get('original_t'), d.get('prompt'), d.get('corrupted'), d.get('logits'),
                                    d.get('prediction'), d.get('inference_time', 0))

    def clone(self):
        return EvaluationResultUnit(self.h, self.r, self.t, self.original_h, self.original_r, self.original_t,
                                    self.prompt, self.corrupted, self.logits, self.prediction)

    @staticmethod
    def from_batch(batch) -> List['EvaluationResultUnit']:
        # batch is {'h': [1, 2, 3], 'r': [1, 2, 3], 't': [1, 2, 3], 'prompt': ['a', 'b', 'c'], 'corrupted': [True, False, True]}
        len_batch = len(batch['h'])
        return [EvaluationResultUnit(int(batch['h'][i]), int(batch['r'][i]), int(batch['t'][i]),
                                     int(batch['original_h'][i]), int(batch['original_r'][i]),
                                     int(batch['original_t'][i]),
                                     str(batch['prompt'][i]), bool(batch['corrupted'][i])) for i in range(len_batch)]

    def __repr__(self):
        return f"({self.h}, {self.r}, {self.t})\n" \
               f"prompt: {self.prompt}\n" \
               f"logits: {self.logits}\n" \
               f"prediction: {self.prediction}\n" \
               f"corrupted: {self.corrupted}\n" \
               f"inference_time: {self.inference_time}"

    def set_evaluation_results(self, logits, prediction, inference_time):
        assert isinstance(logits, torch.Tensor)
        assert isinstance(prediction, torch.Tensor)
        logits = logits.detach().cpu()
        prediction = prediction.detach().cpu().item()
        self.logits = logits
        self.prediction = prediction
        self.inference_time = inference_time


class BatchEvaluationInput(torch.utils.data.Dataset):

    def __init__(self, triples: torch.Tensor, original_triples: torch.Tensor, prompts: List[str],
                 corrupted: torch.Tensor):
        triples = torch.tensor(triples).detach().cpu()
        original_triples = torch.tensor(original_triples).detach().cpu()
        corrupted = torch.tensor(corrupted).detach().cpu().bool()
        assert len(triples) == len(original_triples) == len(prompts) == len(corrupted)

        self.triples = triples
        self.original_triples = original_triples
        self.prompts = prompts
        self.corrupted = corrupted

    @staticmethod
    def from_list(units: List[EvaluationResultUnit]):
        triples = torch.tensor([(unit.h, unit.r, unit.t) for unit in units])
        original_triples = torch.tensor([(unit.original_h, unit.original_r, unit.original_t) for unit in units])
        prompts = [unit.prompt for unit in units]
        corrupted = torch.tensor([unit.corrupted for unit in units])
        return BatchEvaluationInput(triples, original_triples, prompts, corrupted)

    def share_memory(self):
        self.triples.share_memory_()
        self.original_triples.share_memory_()
        if not isinstance(self.prompts, SharedStringList):
            self.prompts = SharedStringList(self.prompts)
        self.prompts.share_memory()
        self.corrupted.share_memory_()

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        return {
            'idx': idx,
            'prompt': self.prompts[idx],
            'corrupted': self.corrupted[idx]
        }

    def get_whole_input(self, idx):
        return {
            'h': self.triples[idx][0],
            'r': self.triples[idx][1],
            't': self.triples[idx][2],
            'original_h': self.original_triples[idx][0],
            'original_r': self.original_triples[idx][1],
            'original_t': self.original_triples[idx][2],
            'prompt': self.prompts[idx],
            'corrupted': self.corrupted[idx]

        }


class BatchEvaluationOutput:
    def __init__(self, idx: torch.Tensor, logits: torch.Tensor, predictions: torch.Tensor, inference_time: float):
        self.logits = logits
        self.predictions = predictions
        self.inference_time = inference_time
        self.idx = idx

    def share_memory(self):
        self.logits.share_memory_()
        self.predictions.share_memory_()
        self.idx.share_memory_()

    def __getitem__(self, idx):
        return {
            'logits': self.logits[idx],
            'prediction': self.predictions[idx],
            'inference_time': self.inference_time
        }

    def get_idx(self, idx):
        return int(self.idx[idx])

    def __len__(self):
        return len(self.predictions)


class QueriesDataset(torch.utils.data.Dataset):
    def __init__(self, queries: List[EvaluationResultUnit] = None, shared_memory_queries=None, using_idx=False):
        self.queries = queries
        self.using_idx = using_idx
        self.shared_memory_queries = shared_memory_queries

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        if self.using_idx:
            return {
                'idx': idx,
                'prompt': self.queries[idx].prompt,
                'corrupted': self.queries[idx].corrupted
            }

        return {
            'h': self.queries[idx].h,
            'r': self.queries[idx].r,
            't': self.queries[idx].t,
            'original_h': self.queries[idx].original_h,
            'original_r': self.queries[idx].original_r,
            'original_t': self.queries[idx].original_t,
            'prompt': self.queries[idx].prompt,
            'corrupted': self.queries[idx].corrupted
        }


def verbose_evaluation_progress(results: List[EvaluationResultUnit], idx_mask=None):
    yes, no, idk = 0, 0, 0
    result_len = len(results)
    if idx_mask is not None:
        result_len = idx_mask.sum().item()
    for i, result in enumerate(results):
        if idx_mask is None or idx_mask[i].item():
            if result.prediction == 0:
                yes += 1
            elif result.prediction == 1:
                no += 1
            elif result.prediction == 2:
                idk += 1
    print("Total:", result_len)
    print(f"TRUE: {yes}, FALSE: {no}, IDK: {idk}")
    # keep the percentage
    print(
        f"TRUE: {yes / (result_len + 1e-8) * 100:.2f}%, FALSE:"
        f" {no / (result_len + 1e-8) * 100:.2f}%, IDK:"
        f" {idk / (result_len + 1e-8) * 100:.2f}%")
    print("Total inference time:", sum([result.inference_time for result in results]), "s")
    print("Average inference time:", sum([result.inference_time for result in results]) / (result_len + 1e-8), "s")
    print("Average speed:", (result_len + 1e-8) / sum([result.inference_time for result in results]), "units/s")
