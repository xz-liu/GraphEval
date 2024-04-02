import torch

from torch.utils.data import Dataset, DataLoader

from sqlitedict import SqliteDict
import io
import os
from utils import *


class DictCache:
    def __init__(self):
        self.cache = {}

    def exist(self, key):
        return key in self.cache

    def add_cache(self, key, value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
        self.cache[key] = value

    def get_cache(self, key):
        return self.cache[key].cuda()

    def save_last(self, *args, **kwargs):
        pass


class HiddenStateCache:
    def __init__(self, path):
        # if path not exist, create folder
        if not os.path.exists(path):
            os.makedirs(path)

        if not path.endswith('.sqlite'):
            path += 'db.sqlite'
        self.db = SqliteDict(path)

    # def __getitem__(self, key):
    #
    #     if not isinstance(key, bytes):
    #         key = bytes(key, encoding='utf-8')
    #
    #     return torch.load(io.BytesIO(self.db.get(key)))
    #
    # def __setitem__(self, key, value):
    #
    #     if not isinstance(key, bytes):
    #         key = bytes(key, encoding='utf-8')
    #     # first, initialize a BytesIO object
    #     f = io.BytesIO()
    #
    #     # then, save the value to the BytesIO object
    #     torch.save(value, f)
    #
    #     # finally, save the BytesIO object to the database
    #     self.db.put(key, f.getvalue())
    #
    # def __contains__(self, key):
    #
    #     if not isinstance(key, bytes):
    #         key = bytes(key, encoding='utf-8')
    #     return key in self.db

    def __del__(self):
        self.db.close()

    def __len__(self):
        return len(self.db)

    def __iter__(self):
        return iter(self.db)

    def __repr__(self):
        return repr(self.db)

    def __str__(self):
        return str(self.db)

    def exist(self, key):

        return key in self.db

    def add_cache(self, key, value):
        print('add cache: ', key)
        self.db[key] = value

    def get_cache(self, key):
        print('get cache: ', key)
        return self.db[key]

    def save_last(self, *args, **kwargs):
        pass


class LLMStateData(Dataset):
    def __init__(self, sentences, llm_model):
        self.sentences = sentences
        self.llm_model = llm_model

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.answer[idx]


from transformers import LlamaTokenizer, LlamaModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, \
    SequenceClassifierOutputWithPast


def customized_llama_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_layers=slice(0, None),
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    seq_length_with_past = seq_length
    past_key_values_length = 0

    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    # embed positions
    if attention_mask is None:
        attention_mask = torch.ones(
            (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
        )
        padding_mask = None
    else:
        if 0 in attention_mask:
            padding_mask = attention_mask
        else:
            padding_mask = None

    attention_mask = self._prepare_decoder_attention_mask(
        attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
    )

    hidden_states = inputs_embeds

    if self.gradient_checkpointing and self.training:
        if use_cache:
            # logger.warning_once(
            #     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            # )
            use_cache = False

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    for idx, decoder_layer in enumerate(self.layers[use_layers]):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        if self.gradient_checkpointing and self.training:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs, past_key_value, output_attentions, padding_mask=padding_mask)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(decoder_layer), hidden_states, attention_mask, position_ids
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                padding_mask=padding_mask,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def combine_hidden_states_forwarding(left_sentences, right_sentences, model, tokenizer, prefix_layers=slice(0, 16),
                                     suffix_layers=slice(16, None), device='cuda', left_cache=None):
    # tuple of 33 [[batch, seq, hidden], [batch, seq, hidden], ...]
    if not isinstance(model, LlamaModel) and isinstance(model.model, LlamaModel):
        model = model.model
    if left_cache is not None:
        max_len =max( [x.shape[1] for x in left_cache])
        padding = torch.zeros(len(left_cache), max_len, left_cache[0].shape[2], device=device)
        attn_masks = torch.zeros(len(left_cache), max_len, device=device)
        for i, x in enumerate(left_cache):
            padding[i, :x.shape[1], :] = x.to(device)
            attn_masks[i, :x.shape[1]] = 1
        left_outputs = BaseModelOutputWithPast(last_hidden_state=padding, past_key_values=None ,
                                               hidden_states=None, attentions=None)
        left_attn_mask = attn_masks

    else:
        left_inputs = tokenizer(left_sentences, return_tensors="pt", max_length=256, padding=True, truncation=True)
        left_inputs = {k: v.to(device) for k, v in left_inputs.items()}
        left_outputs = customized_llama_forward(model, **left_inputs, use_layers=prefix_layers)
        left_attn_mask = left_inputs['attention_mask']

    right_inputs = tokenizer(right_sentences, return_tensors="pt", max_length=256, padding=True, truncation=True)
    right_inputs = {k: v.to(device) for k, v in right_inputs.items()}
    # right_position_ids = left_inputs['input_ids'].shape[1] + torch.arange(0, right_inputs['input_ids'].shape[1],
    #                                                                       dtype=torch.long,
    #                                                                       device=left_inputs['input_ids'].device)

    right_position_ids = None
    right_outputs = customized_llama_forward(model, **right_inputs, use_layers=prefix_layers,
                                             position_ids=right_position_ids)
    last_hidden_states = torch.cat(
        [left_outputs.last_hidden_state, right_outputs.last_hidden_state[:, 1:, :]], dim=1)
    # [batch, seq, hidden]
    att_masks = torch.cat([left_attn_mask, right_inputs['attention_mask'][:, 1:]], dim=1)

    return customized_llama_forward(model, inputs_embeds=last_hidden_states, attention_mask=att_masks,
                                    use_layers=suffix_layers, output_hidden_states=True)
