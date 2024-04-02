import torch
from tqdm import tqdm
from args import args, supported_kgs
from collections import defaultdict
import random
import os
import argparse
from torch import Tensor
from utils import *
from os.path import join as pjoin
import pickle
from typing import Tuple, Optional, List, Dict, Any, Union, Set
from transformers import AutoModel, AutoTokenizer, AutoConfig


def relation_url_to_name(url, which):
    return entity_url_to_name(url, which)


def entity_url_to_name(url, which):
    # decode url to name
    # <http://dbpedia.org/resource/!!!> -> !!!
    # TODO ONLY WORKS FOR DBPEDIA ENGLISH
    assert which in supported_kgs
    if which == 'umls':
        return url
    url = bytes(url, 'utf-8').decode('unicode_escape')
    url = url.split('/')[-1][:-1]
    url = url.split('#')[-1]
    url = url.replace('_', ' ')

    return url


class KG:
    def __init__(self, which, triples, entities: set, relations: set):

        assert which in supported_kgs
        self.which = which
        # indexing the entities and relations
        self.entity2id = {}
        self.id2entity = {}
        self.relation2id = {}
        self.id2relation = {}

        for i, entity in enumerate(entities):
            self.entity2id[entity] = i
            self.id2entity[i] = entity

        for i, relation in enumerate(relations):
            self.relation2id[relation] = i
            self.id2relation[i] = relation

        # indexing the triples
        self.triples = []
        for h, r, t in triples:
            h_id = self.entity2id[h]
            r_id = self.relation2id[r]
            t_id = self.entity2id[t]
            self.triples.append((h_id, r_id, t_id))

        # entity url to name
        self.entitynames = []
        for entity in self.id2entity.values():
            self.entitynames.append(entity_url_to_name(entity, which))

        # relation url to name
        self.relationnames = []
        for relation in self.id2relation.values():
            self.relationnames.append(relation_url_to_name(relation, which))

        self.hr2t = defaultdict(set)
        self.r2t = defaultdict(set)
        for h, r, t in self.triples:
            self.hr2t[(h, r)].add(t)
            self.r2t[r].add(t)

    def provide_qa_pair(self, batch: torch.Tensor):
        # batch is [batch_size, 3]
        batch = batch.cpu().numpy()
        template = "What/Who/Where is the {} of {}?"
        questions = []
        answers = []
        for h, r, t in batch:
            questions.append(template.format(self.relationnames[r], self.entitynames[h]))
            answers.append(self.entitynames[t])
        return questions, answers

    def tail_option_ids(self, h, r, option_size):

        r2t = set(self.r2t[r])
        hr2t = set(self.hr2t[(h, r)])

        options = random.sample(sorted(r2t - hr2t), min(len(r2t - hr2t), option_size))

        # Ensure we have enough options
        while len(options) < option_size:
            t = random.choice(list(self.id2entity.keys()))
            if t not in options:
                options.append(t)
        return options

    def provide_tail_options(self, h: int, r: int, target: int, option_size: int = 4,
                             option_hook=None):
        if option_hook is None:
            option_hook = lambda head, tail: tail
        # Select option_size - 1 random elements excluding hr2t
        if option_size > 1:
            options = self.tail_option_ids(h, r, option_size - 1)
        else:
            options = []

        options.append(target)

        random.shuffle(options)

        idx = options.index(target)
        # Generating option prompt
        option_prompt = ''
        for i in range(len(options)):
            # A: Beijing
            option_prompt += chr(ord('A') + i) + ': ' + option_hook(head=self.entitynames[h],
                                                                    tail=self.entitynames[options[i]]) + '\n'
        # answer = chr(ord('A') + idx)

        # option_prompt = "A: {}\nB: {}\nC: {}\nD: {}\nE: None of the above or I don't know\n".format(
        #     *(option_hook(head=self.entitynames[h], tail=self.entitynames[x]) for x in options))
        answer = chr(ord('A') + idx)
        # print(option_prompt)
        return option_prompt, answer, options

    def __getitem__(self, index):
        return self.triples[index]

    def __len__(self):
        return len(self.triples)

    def process_relation_types(self, entity_type_path):
        entity_type_kg = load_dbpedia_kg(entity_type_path)
        entity_type_dict = defaultdict(list)
        # type_set = set()
        # appear_time = defaultdict(int)
        for h, r, t in tqdm(entity_type_kg.triples, desc='Processing entity types'):
            h_url, t_url = entity_type_kg.id2entity[h], entity_type_kg.id2entity[t]

            # filter entity type, only keep schema.org
            if 'schema.org' not in t_url.lower():
                continue
            # remove schema.org <http://schema.org/Person> -> Person
            t_url = t_url.split('/')[-1][:-1]

            entity_type_dict[h_url].append(t_url)
            # type_set.add(t_url)
            # appear_time[(h_url, t_url)] += 1

        print(len(entity_type_dict))

        relation_head = defaultdict(lambda: defaultdict(int))
        relation_tail = defaultdict(lambda: defaultdict(int))
        for h, r, t in tqdm(self.triples):
            h_url, t_url = self.id2entity[h], self.id2entity[t]
            for entity_type in entity_type_dict[h_url]:
                relation_head[r][entity_type] += 1
            for entity_type in entity_type_dict[t_url]:
                relation_tail[r][entity_type] += 1

        relation_types = {}
        # for all relations
        for r in (self.id2relation.keys()):
            # top 1 head type, top 1 tail type
            head_list = sorted(relation_head[r].items(), key=lambda x: x[1], reverse=True)[:1] + [(None, None)]
            tail_list = sorted(relation_tail[r].items(), key=lambda x: x[1], reverse=True)[:1] + [(None, None)]
            relation_types[r] = (head_list[0][0], tail_list[0][0])
            # relation_types[r] =

        for r in relation_types:
            print(
                f"Relation {self.relationnames[r]} head type {relation_types.get(r, (None, None))[0]} tail type {relation_types.get(r, (None, None))[1]}")

            print('-----------------------------------')
        return relation_types


def load_tsv_kg(path):
    triples, entities, relations = [], set(), set()
    with open(path, 'r') as f:
        for line in tqdm(f.readlines()):
            # tr= line.split('\t')
            h, r, t = line.split('\t')
            triples.append((h, r, t))
            entities.add(h)
            entities.add(t)
            relations.add(r)

    return KG(args.which, triples, entities, relations)


def load_dbpedia_kg(path, which=None):
    if which is None:
        which = args.which

    triples = []
    entities = set()
    relations = set()
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            # <http://dbpedia.org/resource/!!!> <http://dbpedia.org/ontology/bandMember> <http://dbpedia.org/resource/Nic_Offer> .
            triple = line.strip().split()
            if len(triple) != 4:
                continue
            h, r, t = triple[0], triple[1], triple[2]

            triples.append((h, r, t))
            entities.add(h)
            entities.add(t)
            relations.add(r)
    print("load dbpedia kg done, triples: %d, entities: %d, relations: %d" % (
        len(triples), len(entities), len(relations)))
    if which.endswith('filtered'):
        rel_names = set()
        for rel in relations:
            rel_names.add(relation_url_to_name(rel, which))
        new_entities = set()
        new_triples = []
        # filter out dummy entity
        for entity in tqdm(list(entities)):
            if "__" in entity:
                # last "__" is the separator
                splitted = entity[:-1].split("__")
                if splitted[-1].isdigit():
                    # print('Entity: %s is dummy entity, removed' % entity)
                    continue
                for sp in splitted:
                    if sp in rel_names:
                        # print('Entity: %s is dummy entity, removed' % entity)
                        continue
            # print("Entity: %s is not dummy entity, kept" % entity)
            new_entities.add(entity)

        for h, r, t in triples:
            if h in new_entities and t in new_entities:
                new_triples.append((h, r, t))

        print("filter dummy entities done, triples: %d, entities: %d, relations: %d" % (
            len(new_triples), len(new_entities), len(relations)))
        return KG(which, new_triples, new_entities, relations)

    # TODO complete filter!!!!!!!!!! IMPORTANT!!!!!!

    return KG(which, triples, entities, relations)


from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F


# os.environ['CUDA_VISIBLE_DEVICES'] = '6'


def get_KG_embeddings(args: argparse.Namespace,
                      no_embeddings=False,
                      invalidate_cache=False) \
        -> Tuple[KG, Union[Tensor, None], Union[Tensor, None]]:
    embedding_dir = args.embedding_dir
    # TODO legend embedding code
    tokenizer = 'bert-base-uncased'
    model = 'bert-base-uncased'
    embedding_batch_size = 512
    layer = -1
    which = args.which
    data_path = args.kg_path
    if not os.path.exists(pjoin(embedding_dir, which)):
        os.makedirs(pjoin(embedding_dir, which), exist_ok=True)

    if not invalidate_cache and os.path.exists(pjoin(embedding_dir, which, 'kg_final')):
        print("loading KG from cache...")
        with open(pjoin(embedding_dir, which, 'kg_final'), 'rb') as f:
            kg = pickle.load(f)
    else:
        print("loading KG...")
        if which in ['dbpedia-en', 'dbpedia-en-filtered']:
            kg = load_dbpedia_kg(data_path)
        elif which in ['umls']:
            kg = load_tsv_kg(data_path)
        else:
            raise NotImplementedError()
        with open(pjoin(embedding_dir, which, 'kg_final'), 'wb') as f:
            pickle.dump(kg, f)
    if no_embeddings:
        return kg, None, None
    # try to load from cache
    # TODO, invalidate cache did not invalidate the embeddings
    cache_path = pjoin(embedding_dir, which.replace('-filtered', ''), 'embeddings.pt')
    if not invalidate_cache and os.path.exists(cache_path):
        print("loading KG embeddings from cache...")
        embeddings, rel_embeddings, entity2id, relation2id = torch.load(cache_path)
        current_entity2id = kg.entity2id
        current_relation2id = kg.relation2id
        # reindex embeddings
        new_embeddings = torch.zeros(len(current_entity2id), embeddings.shape[1])
        new_rel_embeddings = torch.zeros(len(current_relation2id), rel_embeddings.shape[1])
        random_entity_cnt = 0
        random_relation_cnt = 0
        for entity, idx in current_entity2id.items():
            if entity in entity2id:
                new_embeddings[idx] = embeddings[entity2id[entity]]
            else:
                new_embeddings[idx] = torch.randn(embeddings.shape[1])
                random_entity_cnt += 1
        for relation, idx in current_relation2id.items():
            if relation in relation2id:
                new_rel_embeddings[idx] = rel_embeddings[relation2id[relation]]
            else:
                new_rel_embeddings[idx] = torch.randn(rel_embeddings.shape[1])
                random_relation_cnt += 1
        embeddings = new_embeddings
        rel_embeddings = new_rel_embeddings
        print('load embeddings from cache, random entity cnt: {}, random relation cnt: {}'.format(random_entity_cnt,
                                                                                                  random_relation_cnt))
        return kg, embeddings, rel_embeddings

    # load model
    print("loading model...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    model = AutoModel.from_pretrained(model, output_hidden_states=True).cuda()
    model.eval()

    # get embeddings
    print("getting embeddings...")
    embeddings = []
    # kg.entitynames = ['USA', 'China', 'Japan', 'Beijing', 'Shanghai', 'Tokyo']

    with torch.no_grad():
        for i in tqdm(range(0, len(kg.entitynames), embedding_batch_size)):
            batch = kg.entitynames[i:i + embedding_batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to('cuda')
            outputs = model(**inputs)
            if layer >= 0:
                # apply max pooling over the sequence
                embeddings.append(torch.max(outputs.last_hidden_state[:, layer, :], dim=1)[0].cpu())
            else:
                embeddings.append(outputs.pooler_output.cpu())
            torch.cuda.empty_cache()

    # kg.relationnames = ['capital', 'capital', 'capital', 'capital', 'capital', 'capital']

    rel_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(kg.relationnames), embedding_batch_size)):
            batch = kg.relationnames[i:i + embedding_batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to('cuda')
            outputs = model(**inputs)
            if layer >= 0:
                # apply max pooling over the sequence
                embeddings.append(torch.max(outputs.last_hidden_state[:, layer, :], dim=1)[0].cpu())
            else:
                rel_embeddings.append(outputs.pooler_output.cpu())
            torch.cuda.empty_cache()

    # concatenate
    embeddings = torch.cat(embeddings, dim=0)
    rel_embeddings = torch.cat(rel_embeddings, dim=0)

    # save to cache
    print("saving KG embeddings to cache...")
    torch.save((embeddings, rel_embeddings, kg.entity2id, kg.relation2id), cache_path)

    return kg, embeddings, rel_embeddings


def get_kg(args, invalidate_cache=False):
    return get_KG_embeddings(args, no_embeddings=True, invalidate_cache=invalidate_cache)[0]


if __name__ == '__main__':
    pass
    # load_dbpedia_kg('./kg/dbpedia/outs/mappingbased-objects_lang=en.ttl.bzip2.out')
