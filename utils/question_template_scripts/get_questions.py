import torch
from args import *
from preprocess_embeddings import get_KG_embeddings
from collections import defaultdict
from typing import List
from llama import Dialog
from language_models import ChatGPT
from os.path import join as pjoin
import os
import time


def relation_url_to_name(url, which):
    return entity_url_to_name(url, which)


def entity_url_to_name(url, which):
    # decode url to name
    # <http://dbpedia.org/resource/!!!> -> !!!
    # TODO ONLY WORKS FOR DBPEDIA ENGLISH
    assert which in supported_kgs
    url = bytes(url, 'utf-8').decode('unicode_escape')
    url = url.split('/')[-1][:-1]
    url = url.split('#')[-1]
    url = url.replace('_', ' ')

    return url


def timeout_wrapper(func, args, kwargs, timeout_duration=10, default=None):
    import signal

    class TimeoutError(Exception):
        pass

    def handler(signum, frame):
        raise TimeoutError()

    # set the timeout handler
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout_duration)
    try:
        result = func(*args, **kwargs)
    except TimeoutError as exc:
        result = default
    finally:
        signal.alarm(0)
    if result is None:
        # throw error
        raise TimeoutError()

    return result


if __name__ == '__main__':
    # llm_answers_cache = torch.load('./cache/llm_answer_cache.pt')
    # print(len(llm_answers_cache))
    kg, embeddings, rel_embeddings = get_KG_embeddings(args)
    relation_examples = defaultdict(list)

    chatgpt = ChatGPT(openai_key='openai_key')
    llm_question_cache = {}
    cache_path = pjoin(args.answer_cache_dir, 'llm_question_cache_final.pt')
    if os.path.exists(cache_path):
        print("loading llm question cache...")
        llm_question_cache = torch.load(cache_path)
    else:
        os.makedirs(args.answer_cache_dir, exist_ok=True)

    for h, r, t in kg.triples:
        if len(relation_examples[r]) < 10:
            relation_examples[r].append((h, t))

    print('entity size:', len(kg.id2entity))
    print('relation size:', len(kg.id2relation))

    errors = []
    from collections import Counter

    for r, question in llm_question_cache.items():
        if '{head}' not in question:
            errors.append(r)
        elif Counter(question)['{'] != 1:
            errors.append(r)
        elif Counter(question)['}'] != 1:
            errors.append(r)

    print(errors)

    prompt = '''You are given a few samples of a relation in the format of <head, relation, tail>.
You need to write a question template about the relation, which will make a choice about the tail given the head and the relation.
I will give you 3 examples as below.

relation: "education"
triple samples:
<Mamphono Khaketla, education, National University of Lesotho>
<2016 New York and New Jersey bombings, education, Edison High School (New Jersey)>
<2017 Fresno shootings, education, American River College>
<2017 Fresno shootings, education, Cosumnes River College>
<2017 Fresno shootings, education, Fresno City College>
<2017 Fresno shootings, education, Sacramento City College>
<A'Lelia Bundles, education, Columbia University Graduate School of Journalism>
<A'Lelia Bundles, education, Harvard College>
<A.C. Jackson, education, Meharry Medical College>
<A.D. Frazier, education, University of North Carolina at Chapel Hill>

question template: "Where is {head} educated at?"


relation: "channel"
triple samples:
<Way Out, channel, CBS>
<100 Code, channel, Sky Krimi>
<100 dÃ­as para enamorarse (Argentine TV series), channel, Telefe>
<12 O'Clock High (TV series), channel, American Broadcasting Company>
<12 Signs of Love, channel, TVN (South Korean TV channel)>
<12 Years Promise, channel, JTBC>
<12 Years Promise, channel, OBS Gyeongin TV>
<13 Commandments, channel, VTM (TV channel)>
<18 Wheels Across America, channel, Discovery Channel (Polish TV channel)>
<19+, channel, TVN (Poland)>

question template: "Which channel is {head} broadcast on?"


relation "curator"
triple samples:
<Alicante Museum of Contemporary Art, curator, Alicante>
<Angelika Kauffmann Museum, curator, Bettina BaumgÃ¤rtel>
<Archaeological Museum of Alicante, curator, Alicante (province)>
<Archaeology Museum, Sogamoso, curator, Tunja>
<Archaeology Museum, Sogamoso, curator, Universidad PedagÃ³gica y TecnolÃ³gica de Colombia>
<Art Gallery of Alberta, curator, Catherine Crowston>
<Bailey-Matthews National Shell Museum, curator, JosÃ© H. Leal>
<Bangabandhu Military Museum, curator, Bangladesh Army>
<Bardo National Museum (Tunis), curator, Moncef Ben Moussa>
<Baturyn Museum of Archeology, curator, Hetman's Capital>

question template: "Who is the curator of {head}?"

'''

    asked_question_prompt = '''
according to the above 3 examples, please write a question template for the folloing relation:

relation: "{relation}"
triple samples:
{triples}

please help me to write a question template for this relation, just give me the question template (no other information like question template:).
'''

    with open('../relations_clean3.txt', 'w') as f:
        for r, relation in kg.id2relation.items():
            relation = relation_url_to_name(relation, args.which)
            triples = []
            for h, t in relation_examples[r]:
                triples.append(
                    f'<{entity_url_to_name(kg.id2entity[h], args.which)}, {relation}, {entity_url_to_name(kg.id2entity[t], args.which)}>')
            triples = '\n'.join(triples)

            context = prompt + asked_question_prompt.format(relation=relation, triples=triples)
            dialogs: List[Dialog] = [
                [
                    {
                        "role": "user",
                        "content": context
                    }
                ]
            ]

            if r in llm_question_cache:
                print(f'{relation} already in cache')
                continue

            # use timeout wrapper to avoid waiting too long
            # question_template = chatgpt.chat_completion(dialogs)[0]
            while True:
                try:
                    question_template = timeout_wrapper(chatgpt.chat_completion, (dialogs,), {})
                    break
                except:
                    print('timeout, retrying...')
                    # sleep(5)
                    time.sleep(5)
            llm_question_cache[r] = question_template[0]

            torch.save(llm_question_cache, cache_path)
            f.write(f'{r}\t{relation}\n')
            for h, t in relation_examples[r]:
                f.write(
                    f'<{entity_url_to_name(kg.id2entity[h], args.which)}, {relation}, {entity_url_to_name(kg.id2entity[t], args.which)}>\n')
            f.write(f'question template: {question_template[0]}\n\n')