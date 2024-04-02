import os.path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformers import AdamW
from torch.optim import Adam
import torch.nn.functional as F

from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig, AutoModelForCausalLM, AutoTokenizer, \
    AutoModel, AutoConfig, AutoModelForSequenceClassification

import utils
from .llm_state_loader import HiddenStateCache, DictCache
from transformers.modeling_outputs import SequenceClassifierOutput
from args import args
from utils import *
from peft import PromptEncoder, PromptEncoderConfig, get_peft_config, get_peft_model, PeftModelForCausalLM, PeftModel


class IdxSelMLP(nn.Module):
    def __init__(self, layers=[256, 128, 64], options=2):
        super(IdxSelMLP, self).__init__()
        weights = []
        biases = []
        for i in range(len(layers) - 1):
            weights.append(nn.Parameter(torch.randn(options, layers[i], layers[i + 1])))
            biases.append(nn.Parameter(torch.randn(options, layers[i + 1])))
        self.weights = nn.ParameterList(weights)
        self.biases = nn.ParameterList(biases)
        self._init_params()

    def _init_params(self):
        for i in range(len(self.weights)):
            nn.init.xavier_normal_(self.weights[i])
            nn.init.zeros_(self.biases[i])

    def forward(self, x, idx):
        batch_size = x.size(0)
        for i in range(len(self.weights)):
            x = x.view(batch_size, 1, -1)
            x = torch.bmm(x, self.weights[i][idx]) + self.biases[i][torch.zeros_like(idx)].view(batch_size, 1, -1)
            # x = torch.bmm(x, self.weights[i][idx]) + self.biases[i][idx].view(batch_size, 1, -1)
            # x = torch.bmm(x, self.weights[i][idx]) + self.biases[i][idx].view(batch_size, 1, -1)
            # x = F.relu(x)
            # x = F.dropout(x, p=0.5, training=self.training)
        return x.view(batch_size, -1)


class FeedForward(nn.Module):
    def __init__(self, seq_len, emb_size, n_classes=2, true_false_dim=-1):
        super(FeedForward, self).__init__()
        self.seq_len = seq_len
        self.emb_size = emb_size
        if true_false_dim > 0:
            tr = torch.randn(1, true_false_dim)
            fl = -tr
            self.register_buffer('true_false_tensor', torch.cat((tr, fl), dim=0))
            self.network = IdxSelMLP([emb_size * seq_len, 256, 128, 64, n_classes], 2)
        else:
            self.true_false_tensor = None

            true_false_dim = max(0, true_false_dim)
            self.network = nn.Sequential(
                nn.Linear(emb_size * seq_len + true_false_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, n_classes)
            )

    def forward(self, x, which_layer=None, true_false=None, **kwargs):
        if which_layer is not None:
            x = x[:, which_layer, :, :]
        # x = torch.max(x, dim=-2).values
        x = x.view(x.size(0), -1)

        if self.true_false_tensor is not None:
            x = self.network(x, true_false)
        # [batch_size, n_channel, emb_size]
        else:
            x = self.network(x)

        return x


class TFLinear(nn.Module):
    def __init__(self, emb_size, seq_len, n_classes=2, tf_dim=-1):
        super(TFLinear, self).__init__()
        self.dense = nn.Linear(emb_size * seq_len, emb_size * seq_len)
        self.layer_norm = nn.LayerNorm(emb_size * seq_len)
        tf_dim = max(0, tf_dim)
        if tf_dim > 0:
            self.decoder = IdxSelMLP([emb_size * seq_len, n_classes])

    def forward(self, x, true_false=None, **kwargs):
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        x = F.gelu(x)
        x = self.layer_norm(x)
        x = self.decoder(x, true_false)  # + self.bias
        return x


class FineTuneModel(nn.Module):
    def __init__(self, model_name, n_channel, seq_len, emb_size, n_classes=2, device='cuda',
                 has_cache=False, use_cache=False, which_classifier='feedforward', cache_type='dict',
                 llm_infer='forward', keep_layers=slice(1, None, None), keep_seq=slice(-1), keep_seq_len=None,
                 tf_dim=-1, peft_config=None, peft_path=None, trained_clf=None):
        super(FineTuneModel, self).__init__()
        if has_cache:
            self.transformer_model, self.tokenizer, self.config = None, None, None
        else:
            if 'bert' in model_name:
                # TODO
                self.transformer_model = AutoModelForCausalLM.from_pretrained(model_name)
                self.transformer_model.to(device)
            else:
                params = dict(device_map='auto',torch_dtype=torch.float16,use_flash_attention_2= not args.no_flash_attention)
                self.transformer_model = AutoModelForCausalLM.from_pretrained(model_name, **params)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.transformer_model.config.pad_token_id = self.transformer_model.config.eos_token_id
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = 'left'
            self.config = self.transformer_model.config
            if peft_config is not None:
                self.transformer_model = get_peft_model(self.transformer_model, peft_config)
            if peft_path is not None:
                print('load peft model from', peft_path)
                self.transformer_model = PeftModel.from_pretrained(self.transformer_model, peft_path, device_map='auto')
            for params in self.transformer_model.parameters():
                params.requires_grad = False

        self.output_hidden_states = True
        self.output_attentions = False
        if emb_size < 0:
            emb_size = self.config.hidden_size

        if which_classifier == 'feedforward':
            if keep_seq_len is None:
                keep_seq_len = seq_len
            # self.cnn = FeedForward(keep_seq_len, emb_size, n_classes, true_false_dim=tf_dim)
            self.cnn = TFLinear(tf_dim=tf_dim, emb_size=emb_size, seq_len=keep_seq_len, n_classes=n_classes)
            if trained_clf is not None:
                self.cnn = load_from(trained_clf)
        else:
            raise ValueError('Invalid classifier type')
        self.n_classes = n_classes
        self.device = device
        self.use_cache = use_cache
        if use_cache:
            if cache_type == 'dict':
                self.cache = DictCache()
            elif cache_type == 'sqlite':
                self.cache = HiddenStateCache(os.path.join(args.hidden_states_cache_path, model_name))
        self.is_autoencoder = False
        self.seq_len = seq_len
        self.keep_layers = keep_layers
        self.keep_seq = keep_seq
        self.llm_infer = llm_infer
        self.tf_dim = tf_dim
        self.cnn.to(device)

    def forward(self, batch, *args, **kwargs):
        if 'labels' in kwargs:
            labels = kwargs.pop('labels').long().to(self.device)
        else:
            labels = None
        device = self.device
        with torch.no_grad():
            responses = {}
            still_need = []
            for i in range(len(batch)):
                if self.use_cache and self.cache.exist(batch[i]):
                    responses[i] = self.cache.get_cache(batch[i])
                else:
                    still_need.append(i)
            if len(still_need) > 0:
                still_need_batch = [batch[i] for i in still_need]

                if self.llm_infer == 'combine':
                    from .llm_state_loader import combine_hidden_states_forwarding
                    pos = kwargs.get('pos', None)
                    if pos is None:
                        raise ValueError('Pos is required for combined hidden states forwarding')
                    left_sents, right_sents = [], []
                    for i in range(len(still_need_batch)):
                        left, right = still_need_batch[i][:pos[i]], still_need_batch[i][pos[i]:]
                        left_sents.append(left)
                        right_sents.append(right)
                    outputs = combine_hidden_states_forwarding(left_sents, right_sents, self.transformer_model,
                                                               self.tokenizer,
                                                               device=device, prefix_layers=slice(0, 12),
                                                               suffix_layers=slice(12, None))
                    # TODO
                    states = torch.stack(outputs.hidden_states, dim=1)
                else:
                    # still_need_labels = labels[torch.tensor(still_need)] if labels is not None else None
                    inputs = self.tokenizer(still_need_batch, padding=True, truncation=True,
                                            max_length=self.seq_len,
                                            return_tensors="pt")
                    input_ids, attention_mask = inputs["input_ids"].to(device), inputs["attention_mask"].to(device)
                    if self.llm_infer == 'forward':
                        outputs = self.transformer_model(input_ids, attention_mask=attention_mask,
                                                         output_hidden_states=self.output_hidden_states,
                                                         output_attentions=self.output_attentions
                                                         )
                        states = torch.stack(outputs.hidden_states, dim=1)
                    elif self.llm_infer == 'generate':
                        outputs = self.transformer_model.generate(input_ids, attention_mask=attention_mask,
                                                                  output_hidden_states=self.output_hidden_states,
                                                                  output_attentions=self.output_attentions,
                                                                  max_new_tokens=1,
                                                                  min_new_tokens=1, return_dict_in_generate=True
                                                                  )
                        # breakpoint()
                        states = torch.stack(outputs.hidden_states[0], dim=1)
                states = states[:, self.keep_layers, self.keep_seq, :].to(self.device).to(torch.float32)
                # print('States shape is ', states.shape)
                # last_layer_hidden_states = outputs.hidden_states[-1]
                for i in range(len(still_need)):
                    responses[still_need[i]] = states[i]
                    if self.use_cache:
                        self.cache.add_cache(still_need_batch[i],
                                             states[i])

            responses = torch.stack([responses[i] for i in range(len(batch))], dim=0)

        # x = torch.stack(responses, dim=0)
        # x = x.permute(1, 0, 2, 3)
        x = responses
        # print('---X shape is ', x.shape)
        if self.tf_dim > 0:
            logits = self.cnn(x, true_false=kwargs.get('true_false', None))
        else:
            logits = self.cnn(x)

        # logits = self.linear(torch.max(last_layer_hidden_states, dim=1).values)
        if labels is None:
            return SequenceClassifierOutput(
                loss=None,
                logits=logits,
                hidden_states=responses,
                attentions=None,
            )
        # print(self.n_classes, labels.shape, logits.shape)
        if self.n_classes == 1:
            squared_norm = torch.sum(logits ** 2, dim=1)
            loss = torch.mean(squared_norm)
        else:
            loss = F.cross_entropy(logits, labels.long())
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=responses,
            attentions=None
        )

    def transformer_logits(self, batch, label_words=None, *args, **kwargs):
        logits = utils.transformer_logits(batch, self.tokenizer, self.transformer_model, self.seq_len, self.device,
                                          label_words=label_words, *args, **kwargs)
        return logits

    def serving(self):
        pass

    def save_clf(self, path):
        save_to(path, self.cnn, 'torch')


if __name__ == '__main__':
    # Example parameters
    pass
