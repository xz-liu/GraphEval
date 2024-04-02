import random

from utils import *
import numpy as np
from .data_generator import get_labels, get_judge_labels
from args import *

from .llm_clients import AskModel

import json


def prepare_pageview_based_triples(kg, triples=None, pageview_dir=None, num_triples=200, top_entities=10000,
                                   ignore_relations=('seeAlso',)):
    pageviews = load_from(pageview_dir, backend='json')

    # top 50000 pageviews
    top_pageviews = sorted(pageviews.items(), key=lambda x: x[1], reverse=True)[:top_entities]
    top_pageviews = set(map(lambda x: x[0], top_pageviews))

    hot_triples = []
    hot_values = {}
    for h, r, t in kg.triples:
        hx = kg.entitynames[h]
        tx = kg.entitynames[t]

        if hx in top_pageviews and tx in top_pageviews:
            hot_triples.append((h, r, t))
            hot_values[(h, r, t)] = pageviews[hx] + pageviews[tx]

    print("Length of hot triples:", len(hot_triples))
    # max 10000 hot triples as tensor, sorted by hotness
    if ignore_relations is not None:
        ignore_relations = set(ignore_relations)
        print('Ignore relations', ignore_relations)
        hot_triples = filter(
            lambda x: kg.relationnames[x[1]] not in ignore_relations, hot_triples
        )
    hot_triples = torch.tensor(sorted(hot_triples, key=lambda x: hot_values[x], reverse=True)[:num_triples])

    return hot_triples


def prepare_random_triples(kg=None, triples=None, num_triples=50000):
    if kg is None:
        kg, _, _ = get_KG_embeddings(args, no_embeddings=True, invalidate_cache=False)

    if triples is not None:
        triples = torch.tensor(triples)
    else:
        # select num_triples random triples
        triples = torch.tensor(kg.triples)[torch.randperm(len(kg.triples))[:num_triples]]

    return triples


def prepare_triples_based_on_entities(kg, entities=None, entity_sz=100000):
    # random select 100000 entities
    if entities is None:
        entities = np.random.choice(len(kg.entitynames), entity_sz, replace=False)
        entities = set(entities.tolist())
    new_triples = []
    for h, r, t in tqdm(kg.triples):
        h, r, t = int(h), int(r), int(t)
        if h in entities and t in entities:
            new_triples.append((h, r, t))

    triples = torch.tensor(new_triples)

    return triples


def get_training_triples(kg=None, triples=None, model_name=None, model_type='llama',
                         test_case_name='undefined',
                         llm_question_template=None,
                         llm_answer_template=None):
    save_path = training_triples_path(model_name, test_case_name)
    if os.path.exists(save_path):
        print("loading training triples from cache...", save_path)
        return torch.load(save_path)
    if triples is None or kg is None:
        raise ValueError("triples and kg must be provided", 'Model name:', model_name, 'Test case name:',
                         test_case_name, 'Model type:', model_type)

    return preprocess_label(kg, triples, model_name, model_type, llm_question_template, llm_answer_template)


def preprocess_label(kg, triples=None, model_name=None, model_type='llama',
                     test_case_name='undefined',
                     llm_question_template=None,
                     llm_answer_template=None, label_type='judge'):
    save_path = training_triples_path(model_name, test_case_name)

    assert model_type in ['llama', 'gemma','chatgpt']
    # TODO add more model types
    if llm_question_template is None:
        llm_question_template = load_from(kg_question_paths[args.which])
    if llm_answer_template is None:
        llm_answer_template = load_from(kg_answer_paths[args.which])

    ask_llama = AskModel(model_name=model_name, token=hf_token, model_type=model_type)

    assert label_type in ['default', 'judge']
    if label_type == 'default':
        training_data = get_labels(triples, kg, llm_question_cache=llm_question_template,
                                   llm_answer_cache=llm_answer_template, llm_client=ask_llama)
    elif label_type == 'judge':
        training_data = get_judge_labels(triples, kg, llm_question_cache=llm_question_template,
                                         llm_answer_cache=llm_answer_template, llm_client=ask_llama, ask_times=1)
    else:
        raise NotImplementedError(f"label_type {label_type} not implemented")
    save_to(save_path, training_data)
    return training_data


def preprocess_true_false_label(kg, triples, model_name=None, model_type='llama',
                                test_case_name='true_false',
                                llm_question_template=None,
                                llm_answer_template=None):
    '''
    preprocess true false labels, as in SAPLMA paper, the true set is the original triples, and the false set is the
    original triples with a random tail entity, there is no E label

    :param kg:
    :param triples:
    :param model_name:
    :param model_type:
    :param test_case_name:
    :param llm_question_template:
    :param llm_answer_template:
    :return:
    '''
    save_path = training_triples_path(model_name, test_case_name)

    assert model_type in ['llama']
    # TODO add more model types
    # if llm_question_template is None:
    #     llm_question_template = load_from(kg_question_paths[args.which])
    # if llm_answer_template is None:
    #     llm_answer_template = load_from(kg_answer_paths[args.which])
    half = int(len(triples) / 2)
    true_set = triples[:half]
    false_set = []
    for h, r, t in tqdm(triples[half:]):
        h, r, t = h.item(), r.item(), t.item()
        _, answer, options = kg.provide_tail_options(h, r, t, option_hook=None)

        t_false = options[random.choice([i for i in range(len(options)) if i != ord(answer) - ord('A')])]

        false_set.append((h, r, t_false))
    from .data_generator import LLMTrainData
    true_set, false_set = torch.tensor(true_set), torch.tensor(false_set)
    batch = torch.concat([true_set, false_set], 0)
    # random shuffle
    batch = batch[torch.randperm(len(batch))]
    data = LLMTrainData(true_set, false_set, batch, None, ['A'] * len(batch), None)
    save_to(save_path, data)
    return data


def case_exists(test_case_name, model_name, ):
    save_path = training_triples_path(model_name, test_case_name)
    return os.path.exists(save_path)
