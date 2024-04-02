import torch
import numpy as np
from scipy import stats
from args import args
from tqdm import trange
@torch.no_grad()
def sample_and_split(inputs, targets, shot=16, classes=3, balanced_labels=True, only_sample=-1):
    if only_sample < 0 and (classes == 1 or not balanced_labels):
        train_idx = torch.randperm(len(targets))[:shot]
    else:
        # sort by target
        perm = torch.argsort(targets)
        inputs = inputs[perm]
        targets = targets[perm]
        cnts = torch.bincount(targets).cpu()
        shifts = torch.cat([torch.tensor([0]), torch.cumsum(cnts, 0)[:-1]])
        print(cnts, shifts)
        randperms = []
        if only_sample > 0:
            randperms.append(torch.randperm(cnts[0])[:only_sample] + shifts[0])
        else:
            for i in range(classes):
                randperms.append(torch.randperm(cnts[i])[:shot] + shifts[i])
        train_idx = torch.cat(randperms, 0)
    test_idx = torch.ones(len(targets)).bool()
    test_idx[train_idx] = False

    return inputs[train_idx], inputs[test_idx], targets[train_idx], targets[test_idx]


def get_positions(kg, triple, llm_question_cache, llm_answer_cache, model_type=None):
    h, r, t = triple
    # TODO add more type
    from data_gen.llm_clients import add_system_prompt
    from data_gen.data_generator import judge_construct_instruct
    if isinstance(h, torch.Tensor):
        h, r, t = h.item(), r.item(), t.item()
    question_with_options = llm_answer_cache[kg.relationnames[r]]
    # option_str, answer, option_ids = kg.provide_tail_options(h, r, t)
    question_with_options = add_system_prompt(
        judge_construct_instruct(question_with_options, ''), model_type
    )

    pos = min(question_with_options.index('{head}'), question_with_options.index('{tail}'))

    return pos


def transformer_logits(batch,tokenizer, transformer_model, seq_len, device, label_words=None, *args, **kwargs):
    if label_words is None:
        label_words = ['Yes', 'No', 'I']

    # get token id of label words
    yes_token = tokenizer.convert_tokens_to_ids(label_words[0])
    no_token = tokenizer.convert_tokens_to_ids(label_words[1])
    i_token = tokenizer.convert_tokens_to_ids(label_words[2])
    with torch.no_grad():
        inputs = tokenizer(batch, padding=True, truncation=True,
                                max_length=seq_len,
                                return_tensors="pt")
        input_ids, attention_mask = inputs["input_ids"].to(device), inputs["attention_mask"].to(device)
        outputs = transformer_model(input_ids, attention_mask=attention_mask,
                                         )
        logits = outputs.logits
        last_logits = logits[:, -1, :].cpu().clone().detach()

        true_false = kwargs.get('true_false', None)
        yes_logits = last_logits[:, yes_token]
        no_logits = last_logits[:, no_token]
        i_logits = last_logits[:, i_token]
        # print('yes_logits', yes_logits.shape)
        # print('no_logits', no_logits.shape)
        # print('i_logits', i_logits.shape)
        all_logits = torch.stack([yes_logits, no_logits, i_logits], dim=1)
        # print('all_logits', all_logits.shape)
        # all_logits dim: batch_size, 3
        predictions = torch.argmax(all_logits, dim=1)
        # print('predictions', predictions.shape)
        # adjust prediction for true_false
        e_idx = predictions == 2
        predictions[true_false == 0] = 1 - predictions[true_false == 0]
        predictions[e_idx] = 2
        return predictions

def construct_questions(kg, triple, llm_question_cache, llm_answer_cache, ty='llama', add_instructions=False):
    # TODO
    h, r, t = triple
    # TODO add more type
    from data_gen.llm_clients import add_system_prompt
    from data_gen.data_generator import judge_construct_instruct
    if isinstance(h, torch.Tensor):
        h, r, t = h.item(), r.item(), t.item()
    # llm_answer_cache = torch.load(pjoin(args.answer_cache_dir, 'llm_answer_cache_final.pt'))

    # TODO add other types

    # B_INST, E_INST = "[INST]", "[/INST]"
    # question_with_options = "<S>" + B_INST + question + E_INST + llm_answer_cache[kg.relationnames[r]].format(
    #     head=kg.entitynames[h], tail=kg.entitynames[t])
    # + "The answer is " + \
    # kg.entitynames[t])
    # return question, None, None
    question_with_options = llm_answer_cache[kg.relationnames[r]].format(
        head=kg.entitynames[h], tail=kg.entitynames[t])
    if not add_instructions:
        return question_with_options, None, None
    # option_str, answer, option_ids = kg.provide_tail_options(h, r, t)
    question_with_options = add_system_prompt(
        judge_construct_instruct(question_with_options, ''), ty
    )
    return question_with_options, None, None


def logit_estimation(logits_a_dataset1, logits_aprime_dataset1, df, confidence_level=0.95):
    logits_a_dataset1, logits_aprime_dataset1 = \
        np.array(logits_a_dataset1), np.array(logits_aprime_dataset1)
    diff_dataset1 = logits_a_dataset1 - logits_aprime_dataset1

    mean_diff = np.mean(diff_dataset1)
    std_err_diff = np.std(diff_dataset1, ddof=1) / np.sqrt(len(diff_dataset1))

    # df = l
    t_critical = stats.t.ppf((1 + confidence_level) / 2, df)
    margin_of_error = t_critical * std_err_diff
    confidence_interval = (mean_diff - margin_of_error, mean_diff + margin_of_error)

    return confidence_interval


def small_lm_of(model_name=None, model_type=None):
    if args.no_small_lm:
        return model_name
    if args.substitute_model != 'None':
        print('Using a substitute model', args.substitute_model, 'instead of', model_name,'from user args.')
        return args.substitute_model

    if model_type is None:
        assert model_name is not None
        if 'llama' in model_name:
            model_type = 'llama'
        elif 'gemma' in model_name:
            model_type = 'gemma'
        elif 'chatgpt' in model_name:
            model_type = 'chatgpt'
        else:
            raise ValueError(f"model_name {model_name} not implemented")
    if model_type == 'llama':
        print('use small llama', 'meta-llama/Llama-2-7b-chat-hf','instead of', model_name)
        return 'meta-llama/Llama-2-7b-chat-hf'
    elif model_type == 'gemma':
        print('use small gemma', 'google/gemma-2b-it','instead of', model_name)
        return 'google/gemma-2b-it'
    elif model_type=='chatgpt':
        print('use llama', 'meta-llama/Llama-2-7b-chat-hf','instead of', model_name)
        return 'meta-llama/Llama-2-7b-chat-hf'

