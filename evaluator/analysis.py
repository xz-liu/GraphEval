import torch

from .common import *
from args import *

import pandas
from data_gen import LLMTrainData


def estimating_logits(model_name, model_type, test_case_name, kg, hrt2idx, logits_true_triples, logits_false_triples):
    training_triple_data: LLMTrainData = load_from(training_triples_path(model_name, test_case_name, which=kg.which))
    evaluation_model= EvaluationModel.from_path_dict(
        dict(
            model_name=model_name,
            model_type=model_type,
            peft_path=trained_prompt_encoder_path(model_name, test_case_name, which=kg.which),
            classifier_path=trained_clf_path(model_name, test_case_name, which=kg.which),
            model_args={
                'device_map': 'auto',
                'torch_dtype': 'float16',
                'use_flash_attention_2': not args.no_flash_attention
            }
        )
    )


    logits_true_triples = load_from(analysis_path(model_name, test_case_name, which=kg.which, stage='logits_true'))
    logits_false_triples = load_from(analysis_path(model_name, test_case_name, which=kg.which, stage='logits_false'))

# definition of truthfulness
def analysis_on_results(model_name, test_case_name, kg, invalidate_cache=False, pageview_dir=None,
                        entity_types_dir=None):
    result_batch_files = all_evaluation_results(model_name, test_case_name, which=kg.which)
    all_evaluated_triples = kg.triples
    hrt2idx = {(int(h), int(r), int(t)): i for i, (h, r, t) in enumerate(all_evaluated_triples)}
    results_true_triples = [-1 for _ in range(len(all_evaluated_triples))]
    results_false_triples = [[] for _ in range(len(all_evaluated_triples))]
    logits_true_triples = [None for _ in range(len(all_evaluated_triples))]
    logits_false_triples = [[] for _ in range(len(all_evaluated_triples))]
    # 0 true, 1 false, 2 IDK
    first_stage_path = analysis_path(model_name, test_case_name, which=kg.which, stage='results')
    if not invalidate_cache and os.path.exists(first_stage_path):
        results_true_triples, results_false_triples = load_from(
            first_stage_path)
        # logits_true_triples = load_from(analysis_path(model_name, test_case_name, which=kg.which, stage='logits_true'))
        # logits_false_triples = load_from(analysis_path(model_name, test_case_name, which=kg.which, stage='logits_false'))
        # , logits_true_triples, logits_false_triples
    else:
        for result_batch_file in tqdm(result_batch_files):
            curr_batch = load_from(result_batch_file)
            for result in curr_batch:
                h, r, t = result.original_h, result.original_r, result.original_t
                idx = hrt2idx[(h, r, t)]
                if result.corrupted:
                    results_false_triples[idx].append(result.prediction)
                    logits_false_triples[idx].append(np.array(result.logits))
                else:
                    results_true_triples[idx] = result.prediction
                    logits_true_triples[idx] = np.array(result.logits)

        save_to(first_stage_path,
                (results_true_triples, results_false_triples))
        save_to(analysis_path(model_name, test_case_name, which=kg.which, stage='logits_true'), logits_true_triples)
        save_to(analysis_path(model_name, test_case_name, which=kg.which, stage='logits_false'), logits_false_triples)


    combined_results_truthful = [0 for _ in range(len(all_evaluated_triples))]
    combined_results_informative = [0 for _ in range(len(all_evaluated_triples))]
    combined_results_correct = [0 for _ in range(len(all_evaluated_triples))]
    tensor_results_true_triples = torch.tensor([x for x in results_true_triples if x != -1])
    print('result_true=0', (tensor_results_true_triples == 0).sum().item() / (len(tensor_results_true_triples) + 1e-8),
          (tensor_results_true_triples == 0).sum().item())
    print('result_true=1', (tensor_results_true_triples == 1).sum().item() / (len(tensor_results_true_triples) + 1e-8),
          (tensor_results_true_triples == 1).sum().item())
    print('result_true=2', (tensor_results_true_triples == 2).sum().item() / (len(tensor_results_true_triples) + 1e-8),
          (tensor_results_true_triples == 2).sum().item())
    result_falses = []
    for i in range(len(results_false_triples)):
        result_falses.extend(results_false_triples[i])
    tensor_results_false_triples = torch.tensor(result_falses)
    print('result_false=0',
          (tensor_results_false_triples == 0).sum().item() / (len(tensor_results_false_triples) + 1e-8),
          (tensor_results_false_triples == 0).sum().item())
    print('result_false=1',
          (tensor_results_false_triples == 1).sum().item() / (len(tensor_results_false_triples) + 1e-8),
          (tensor_results_false_triples == 1).sum().item())
    print('result_false=2',
          (tensor_results_false_triples == 2).sum().item() / (len(tensor_results_false_triples) + 1e-8),
          (tensor_results_false_triples == 2).sum().item())

    for i in trange(len(all_evaluated_triples)):

        if results_true_triples[i] == 0:
            combined_results_truthful[i] = 1
            combined_results_informative[i] = 1
            combined_results_correct[i] = 1
        elif results_true_triples[i] == 1:
            combined_results_truthful[i] = 0
            combined_results_informative[i] = 1
            combined_results_correct[i] = 0
        else:
            combined_results_truthful[i] = 1
            combined_results_informative[i] = 0
            combined_results_correct[i] = 0

        curr_neg_samples = len(results_false_triples[i])
        score = 1 / (curr_neg_samples + 1e-8)
        for j in range(len(results_false_triples[i])):
            if results_false_triples[i][j] == 0:
                combined_results_truthful[i] -= score
                combined_results_correct[i] -= score
            elif results_false_triples[i][j] == 1:
                continue
            else:
                combined_results_informative[i] -= score
                combined_results_correct[i] -= score
        combined_results_informative[i] = max(0, combined_results_informative[i])
        combined_results_truthful[i] = max(0, combined_results_truthful[i])
        combined_results_correct[i] = max(0, combined_results_correct[i])
        # print(results_true_triples[i], results_false_triples[i], score)

    print('truthful', sum(combined_results_truthful) / len(combined_results_truthful))
    print('informative', sum(combined_results_informative) / len(combined_results_informative))
    print('correct', sum(combined_results_correct) / len(combined_results_correct))

    save_to(analysis_path(model_name, test_case_name, which=kg.which, stage='combined'),
            (combined_results_truthful, combined_results_informative, combined_results_correct))

    # edge_betweenness_centrality
    import networkx as nx
    import matplotlib.pyplot as plt
    G = nx.DiGraph()
    for h, r, t in tqdm(all_evaluated_triples, desc='building graph'):
        G.add_edge(h, t)
    # print('betweenness_centrality')
    # edge_betweenness_centrality = nx.edge_betweenness_centrality(G, backend='cugraph')
    #
    # print('betweenness_centrality done!')
    # edge_betweenness_centrality = {k: v for k, v in
    #                                sorted(edge_betweenness_centrality.items(), key=lambda item: item[1], reverse=True)}
    # # triplet_betweenness_centrality
    # triplet_betweenness_centrality = {}
    # for h, r, t in tqdm(all_evaluated_triples, desc='building triplet_betweenness_centrality'):
    #     triplet_betweenness_centrality[(h, r, t)] = edge_betweenness_centrality[(h, t)]
    # triplet_betweenness_centrality = {k: v for k, v in
    #                                   sorted(triplet_betweenness_centrality.items(), key=lambda item: item[1],
    #                                          reverse=True)}
    # print('triplet_betweenness_centrality done!')
    #
    # save_to(analysis_path(model_name, test_case_name, which=kg.which, stage='betweenness_centrality'),
    #         (edge_betweenness_centrality, triplet_betweenness_centrality))

    node_degrees = G.degree()
    # aggregate the node degrees to triples, for each triple, take the average of the degrees of the head and tail
    triple_degrees = []
    triple_pageviews = []

    assert pageview_dir is not None
    pageviews = load_from(pageview_dir, backend='json')

    for h, r, t in tqdm(all_evaluated_triples, desc='building triple_degrees'):
        triple_degrees.append((node_degrees[h] + node_degrees[t]) / 2.)
        triple_pageviews.append((pageviews.get(kg.entitynames[h], 0) + pageviews.get(kg.entitynames[t], 0)) / 2.)

    triple_names = ['\t'.join([kg.entitynames[h], kg.relationnames[r], kg.entitynames[t]]) for h, r, t in
                    all_evaluated_triples]

    triplename2truthful = {triple_names[i]: combined_results_truthful[i] for i in range(len(triple_names))}
    triplename2informative = {triple_names[i]: combined_results_informative[i] for i in range(len(triple_names))}
    triplename2correct = {triple_names[i]: combined_results_correct[i] for i in range(len(triple_names))}
    triplename2degree = {triple_names[i]: triple_degrees[i] for i in range(len(triple_names))}
    triplename2pageviews = {triple_names[i]: triple_pageviews[i] for i in range(len(triple_names))}
    df = pandas.DataFrame(
        {'truthful': triplename2truthful, 'informative': triplename2informative, 'degree': triplename2degree,
         'pageviews': triplename2pageviews, 'correct': triplename2correct})
    df.to_csv(analysis_path(model_name, test_case_name, which=kg.which, stage='triple_results', format='csv'), sep='\t')
    # df.t
    print(df.corr())
    # aggregate the results to relations
    tid2rel = [int(r) for h, r, t in all_evaluated_triples]

    rel2results_truthful = {}
    rel2results_informative = {}
    rel2results_correct = {}
    for i in trange(len(all_evaluated_triples)):
        r = tid2rel[i]
        if r not in rel2results_truthful:
            rel2results_truthful[r] = []
            rel2results_informative[r] = []
            rel2results_correct[r] = []
        rel2results_truthful[r].append(combined_results_truthful[i])
        rel2results_informative[r].append(combined_results_informative[i])
        rel2results_correct[r].append(combined_results_correct[i])

    for r in tqdm(rel2results_truthful):
        rel2results_truthful[r] = sum(rel2results_truthful[r]) / (len(rel2results_truthful[r]) + 1e-8)
        rel2results_informative[r] = sum(rel2results_informative[r]) / (len(rel2results_informative[r]) + 1e-8)
        rel2results_correct[r] = sum(rel2results_correct[r]) / (len(rel2results_correct[r]) + 1e-8)

    rel_types = kg.process_relation_types(entity_types_dir)
    # aggregate the results to relation types
    headtype2results_truthful = {}
    headtype2results_informative = {}
    headtype2results_correct = {}

    tailtype2results_truthful = {}
    tailtype2results_informative = {}
    tailtype2results_correct = {}

    head_tail_type2results_truthful = {}
    head_tail_type2results_informative = {}
    head_tail_type2results_correct = {}

    for r in tqdm(rel2results_truthful):
        head_type = rel_types[r][0] or 'None'
        tail_type = rel_types[r][1] or 'None'
        if head_type not in headtype2results_truthful:
            headtype2results_truthful[head_type] = []
            headtype2results_informative[head_type] = []
            headtype2results_correct[head_type] = []
        if tail_type not in tailtype2results_truthful:
            tailtype2results_truthful[tail_type] = []
            tailtype2results_informative[tail_type] = []
            tailtype2results_correct[tail_type] = []
        headtype2results_truthful[head_type].append(rel2results_truthful[r])
        headtype2results_informative[head_type].append(rel2results_informative[r])
        headtype2results_correct[head_type].append(rel2results_correct[r])

        tailtype2results_truthful[tail_type].append(rel2results_truthful[r])
        tailtype2results_informative[tail_type].append(rel2results_informative[r])
        tailtype2results_correct[tail_type].append(rel2results_correct[r])

        headType_tailType = '->'.join([head_type, tail_type])
        if headType_tailType not in head_tail_type2results_truthful:
            head_tail_type2results_truthful[headType_tailType] = []
            head_tail_type2results_informative[headType_tailType] = []
            head_tail_type2results_correct[headType_tailType] = []
        head_tail_type2results_truthful[headType_tailType].append(rel2results_truthful[r])
        head_tail_type2results_informative[headType_tailType].append(rel2results_informative[r])
        head_tail_type2results_correct[headType_tailType].append(rel2results_correct[r])

    for r in tqdm(headtype2results_truthful):
        headtype2results_truthful[r] = sum(headtype2results_truthful[r]) / (len(headtype2results_truthful[r]) + 1e-8)
        headtype2results_informative[r] = sum(headtype2results_informative[r]) / (
                len(headtype2results_informative[r]) + 1e-8)
        headtype2results_correct[r] = sum(headtype2results_correct[r]) / (len(headtype2results_correct[r]) + 1e-8)

    for r in tqdm(tailtype2results_truthful):
        tailtype2results_truthful[r] = sum(tailtype2results_truthful[r]) / (len(tailtype2results_truthful[r]) + 1e-8)
        tailtype2results_informative[r] = sum(tailtype2results_informative[r]) / (
                len(tailtype2results_informative[r]) + 1e-8)
        tailtype2results_correct[r] = sum(tailtype2results_correct[r]) / (len(tailtype2results_correct[r]) + 1e-8)

    for r in tqdm(head_tail_type2results_truthful):
        head_tail_type2results_truthful[r] = sum(head_tail_type2results_truthful[r]) / (
                len(head_tail_type2results_truthful[r]) + 1e-8)
        head_tail_type2results_informative[r] = sum(head_tail_type2results_informative[r]) / (
                len(head_tail_type2results_informative[r]) + 1e-8)
        head_tail_type2results_correct[r] = sum(head_tail_type2results_correct[r]) / (
                len(head_tail_type2results_correct[r]) + 1e-8)

    df = pandas.DataFrame({'truthful': headtype2results_truthful, 'informative': headtype2results_informative,
                           'correct': headtype2results_correct})
    df.to_csv(analysis_path(model_name, test_case_name, which=kg.which, stage='head_type_results', format='csv'),
              sep='\t')
    df = pandas.DataFrame({'truthful': tailtype2results_truthful, 'informative': tailtype2results_informative,
                           'correct': tailtype2results_correct})
    df.to_csv(analysis_path(model_name, test_case_name, which=kg.which, stage='tail_type_results', format='csv'),
              sep='\t')
    df = pandas.DataFrame(
        {'truthful': head_tail_type2results_truthful, 'informative': head_tail_type2results_informative,
         'correct': head_tail_type2results_correct})
    df.to_csv(analysis_path(model_name, test_case_name, which=kg.which, stage='head_tail_type_results', format='csv'),
              sep='\t')

    relname2truthful = {kg.relationnames[r]: rel2results_truthful[r] for r in rel2results_truthful}
    relname2informative = {kg.relationnames[r]: rel2results_informative[r] for r in rel2results_informative}
    relname2correct = {kg.relationnames[r]: rel2results_correct[r] for r in rel2results_correct}
    df = pandas.DataFrame(
        {'truthful': relname2truthful, 'informative': relname2informative, 'correct': relname2correct})
    df.to_csv(analysis_path(model_name, test_case_name, which=kg.which, stage='relation_results', format='csv'),
              sep='\t')
