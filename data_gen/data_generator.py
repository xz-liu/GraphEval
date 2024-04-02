import copy

from utils import *

import torch
from tqdm import trange
from .llm_clients import AskModel
from utils.kg_utils import KG
import re


class TripleInfo:
    def __init__(self, h, r, t, kg: KG, asked_question, llm_response, answer_label, real_label):
        self.h = h
        self.r = r
        self.t = t
        self.asked_question = asked_question
        self.llm_response = llm_response
        self.answer_label = answer_label
        self.real_label = real_label
        self.h_name = kg.entitynames[h]
        self.r_name = kg.relationnames[r]
        self.t_name = kg.entitynames[t]


class BatchTripleInfo:
    def __init__(self, kg: KG, triple_info: Dict[Tuple, List[TripleInfo]] = None):
        self.kg = kg
        if triple_info is None:
            triple_info = {}
        self.triple_info = triple_info

    def add_triple_info(self, h, r, t, asked_question, llm_response, answer_label, real_label):
        if (h, r, t) not in self.triple_info:
            self.triple_info[(h, r, t)] = []
        self.triple_info[(h, r, t)].append(
            TripleInfo(h, r, t, self.kg, asked_question, llm_response, answer_label, real_label))

    @staticmethod
    def combine_multiple_asks(triple_info: List['BatchTripleInfo']):
        new_triple_info = {}
        if len(triple_info) == 0:
            return None
        print('combining multiple triple info')
        for info in tqdm(triple_info):
            if info is None:
                continue
            for k, v in info.triple_info.items():
                if k not in new_triple_info:
                    new_triple_info[k] = []
                new_triple_info[k].extend(v)
        return BatchTripleInfo(triple_info[0].kg, new_triple_info)


class LLMTrainData:
    class_correct = 0
    class_incorrect = 1
    class_e = 2

    def __init__(self, positive_triples, negative_triples, batch, options, labels, answer=None, real_labels=None,
                 triple_info=None):
        self.positive_triples = positive_triples
        self.negative_triples = negative_triples
        self.batch = batch
        self.options = options
        self.labels = labels
        self.answer = answer
        self.real_labels = real_labels
        self._preprocess()
        self.triple_info = triple_info

    def get_real_labels(self, batch: torch.Tensor):
        if not hasattr(self, '_batch_hash'):
            self._batch_hash = {}
            for i, triple in enumerate(self.batch):
                h, r, t = map(int, triple)
                self._batch_hash[(h, r, t)] = i
        rl = [self.real_labels[self._batch_hash[(int(h), int(r), int(t))]] for h, r, t in batch]
        rl = torch.all(torch.tensor(rl), dim=-1).long()
        return rl

    @staticmethod
    def constant_judge(multiple_asks: List['LLMTrainData']):
        '''
        constant judge for multiple asks
        TODO
        :param multiple_asks:
        :return:
        '''
        ask_times = len(multiple_asks)
        if ask_times == 1:
            return multiple_asks[0]
        new_batch = multiple_asks[0].batch.clone()
        new_label = []
        new_positive_triples = []
        new_negative_triples = []
        for i in range(len(new_batch)):
            curr_new_label = [multiple_asks[j].labels[i] for j in range(ask_times)]
            # majority vote
            counts = [curr_new_label.count('T'), curr_new_label.count('F'), curr_new_label.count('E')]
            # print(counts)
            h, r, t = new_batch[i]
            h, r, t = int(h), int(r), int(t)

            if counts[0] > counts[1] and counts[0] > counts[2]:
                new_label.append('T')
                new_positive_triples.append([h, r, t])
            elif counts[1] > counts[0] and counts[1] > counts[2]:
                new_label.append('F')
                new_negative_triples.append([h, r, t])
            else:
                new_label.append('E')
        new_positive_triples = torch.tensor(new_positive_triples)
        new_negative_triples = torch.tensor(new_negative_triples)
        new_labels = new_label
        new_options = multiple_asks[0].options
        new_real_labels = multiple_asks[0].real_labels
        new_answer = multiple_asks[0].answer
        new_data = LLMTrainData(new_positive_triples, new_negative_triples, new_batch, new_options, new_labels,
                                new_answer, new_real_labels)
        new_data.triple_info = BatchTripleInfo.combine_multiple_asks([ask.triple_info for ask in multiple_asks])
        return new_data

    @staticmethod
    def majority_vote_judge(multiple_asks: List['LLMTrainData']):
        '''
        majority vote for multiple asks
        :param multiple_asks:
        :return:
        '''
        ask_times = len(multiple_asks)
        if ask_times == 1:
            return multiple_asks[0]
        new_batch = multiple_asks[0].batch.clone()
        new_label = []
        new_positive_triples = []
        new_negative_triples = []
        for i in range(len(new_batch)):
            curr_new_label = [multiple_asks[j].labels[i] for j in range(ask_times)]
            # majority vote
            counts = [curr_new_label.count('T'), curr_new_label.count('F'), curr_new_label.count('E')]
            # print(counts)
            h, r, t = new_batch[i]
            h, r, t = int(h), int(r), int(t)

            if counts[0] > counts[1] and counts[0] > counts[2]:
                new_label.append('T')
                new_positive_triples.append([h, r, t])
            elif counts[1] > counts[0] and counts[1] > counts[2]:
                new_label.append('F')
                new_negative_triples.append([h, r, t])
            else:
                new_label.append('E')
        new_positive_triples = torch.tensor(new_positive_triples)
        new_negative_triples = torch.tensor(new_negative_triples)
        new_labels = new_label
        new_options = multiple_asks[0].options
        new_real_labels = multiple_asks[0].real_labels
        new_answer = multiple_asks[0].answer
        new_data = LLMTrainData(new_positive_triples, new_negative_triples, new_batch, new_options, new_labels,
                                new_answer, new_real_labels)
        new_data.triple_info = BatchTripleInfo.combine_multiple_asks([ask.triple_info for ask in multiple_asks])
        return new_data

    def _all_true(self):
        self.real_labels = [(True, True, True) for _ in range(len(self.batch))]

    def _all_false(self):
        self.real_labels = [(False, False, False) for _ in range(len(self.batch))]

    def _set_real_labels(self, real_labels):
        self.real_labels = real_labels

    def _shuffle(self):
        perm = torch.randperm(len(self.batch))
        # self.positive_triples = self.positive_triples[perm]
        # self.negative_triples = self.negative_triples[perm]
        self.batch = self.batch[perm]
        self.options = [self.options[i] for i in perm]
        self.labels = [self.labels[i] for i in perm]
        self.multi_class_labels = self.multi_class_labels[perm]
        if self.real_labels is not None:
            self.real_labels = [self.real_labels[i] for i in perm]

        self._preprocess()

    def _combine_multiple_asks(self, other: 'LLMTrainData'):
        '''
        merge two LLMTrainData with multiple asks, this means that self.batch == other.batch
        the difference is that self.options and self.labels are different, also the answer is different

        strategy:
        for the same triple, if one is I don't know, then the result is I don't know
        if both are correct, then the result is correct
        if both are incorrect, then the result is incorrect
        if one is correct and the other is incorrect, then the result is incorrect
        :param other:
        :return:
        '''
        assert torch.all(self.batch == other.batch)
        # self.options = [self.options[i] for i in range(len(self.options))]
        self_positive_sets = set([(int(h), int(r), int(t)) for h, r, t in self.positive_triples])
        other_positive_sets = set([(int(h), int(r), int(t)) for h, r, t in other.positive_triples])
        # do not set negative triples
        new_labels = []
        for i in range(len(self.labels)):
            if self.labels[i] == 'E' or other.labels[i] == 'E':
                new_labels.append('E')
            else:
                new_labels.append(self.labels[i])

        positive_sets = self_positive_sets.intersection(other_positive_sets)

        new_positive_triples = torch.tensor(list(positive_sets))
        self.positive_triples = new_positive_triples
        self.labels = new_labels
        self._preprocess()
        self.triple_info = BatchTripleInfo.combine_multiple_asks([self.triple_info, other.triple_info])

    def _merge(self, other: 'LLMTrainData', do_shuffle=False):
        self.positive_triples = torch.cat([self.positive_triples, other.positive_triples], 0)
        self.negative_triples = torch.cat([self.negative_triples, other.negative_triples], 0)
        self.batch = torch.cat([self.batch, other.batch], 0)
        self.options = self.options + other.options
        self.labels = self.labels + other.labels
        self.answer = copy.deepcopy(self.answer)
        self.answer.update(other.answer)
        if self.real_labels is not None and other.real_labels is not None:
            self.real_labels = self.real_labels + other.real_labels
        print('successfull merge')
        if do_shuffle:
            self._shuffle()
        else:
            self._preprocess()

        self.triple_info = BatchTripleInfo.combine_multiple_asks([self.triple_info, other.triple_info])

    def _preprocess(self, reset=True):
        positive_set = set([(int(h), int(r), int(t)) for h, r, t in self.positive_triples])
        correct_labels = [0 for _ in range(len(self.batch))]
        e_labels = [0 for _ in range(len(self.batch))]
        truthful_labels = [0 for _ in range(len(self.batch))]
        informative_labels = [0 for _ in range(len(self.batch))]
        multi_class_labels = [0 for _ in range(len(self.batch))]
        # truthful: correct or E
        # informative: not E
        for i, triple in enumerate(self.batch):
            h, r, t = map(int, triple)
            multi_class_labels[i] = self.class_incorrect
            if (h, r, t) in positive_set:
                correct_labels[i] = 1
                truthful_labels[i] = 1
                multi_class_labels[i] = self.class_correct
            if self.labels[i] != 'E':
                informative_labels[i] = 1
            if self.labels[i] == 'E':
                e_labels[i] = 1
                truthful_labels[i] = 1
                multi_class_labels[i] = self.class_e

        all_labels = tuple(map(torch.tensor,
                               [correct_labels,
                                e_labels,
                                truthful_labels,
                                informative_labels,
                                multi_class_labels]))
        if reset:
            (self.correct_labels,
             self.e_labels,
             self.truthful_labels,
             self.informative_labels,
             self.multi_class_labels) = all_labels
        return all_labels

    def __repr__(self):
        return (
            f'LLMTrainData(total={len(self.batch)}, true={sum(self.correct_labels)},\n\t'
            f' e={sum(self.e_labels)},\n\t'
            f'truthful={sum(self.truthful_labels)},\n\t'
            f'informative={sum(self.informative_labels)},\n\t'
            f'class_correct={torch.sum(self.multi_class_labels == self.class_correct)},\n\t'
            f'class_incorrect={torch.sum(self.multi_class_labels == self.class_incorrect)},\n\t'
            f'class_e={torch.sum(self.multi_class_labels == self.class_e)},\n\t'
            f'truthful_rate={sum(self.truthful_labels) / len(self.truthful_labels)},\n\t'
            f'informative_rate={sum(self.informative_labels) / len(self.informative_labels)},\n\t'
            f'correct_rate={sum(self.correct_labels) / len(self.correct_labels)},\n\t'
            f'e_rate={sum(self.e_labels) / len(self.e_labels)})')
        # print('truthful rate', sum(truthful_labels) / len(truthful_labels))
        # print('informativeness rate', sum(informative_labels) / len(informative_labels))
        # print('correct rate', sum(correct_labels) / len(correct_labels))
        # print('E rate', sum(e_labels) / len(e_labels))


def default_construct_instruct(question, option_str):
    return '''### Instruction:\n Please first select the correct option (A, B, C, D or E) for question above with one character only, then explain your choice.
If you don't know the answer or you think none of the options is correct, select E. Only select an option other than E if you are sure that it is correct.''' + '\n\n' + \
        "### Input: " + question + '\n' + option_str + '\n\n ### Response: '


def judge_construct_instruct(question, option_str):
    return (
            '''### Instruction:\n You are given a statement. You are asked to judge whether the statement is true or false. ''' +
            '''Answer 'Yes, the statement is true.' ''' +
            '''if you know the statement is true. ''' +
            '''Answer 'No, the statement is false.' ''' +
            '''if you know the statement is false. ''' +
            '''Otherwise, answer 'I don't know.' ''' +
            # '''If you do not know whether the statement is true, answer "I don't know".''' +
            '\n\n' +
            "### Input: " + question)


def judge_label_transform(answer, false_statement=False):
    print('-----------------')
    print('Set answer: ', answer)
    answer = re.sub(r'[^\w\s]', '', answer)
    answer = answer.split()
    answer = set(answer)
    final_label = 'E'
    if 'true' in answer or 'Yes' in answer:
        if false_statement:
            final_label = 'F'
        else:
            final_label = 'T'
    elif 'false' in answer or 'No' in answer:
        if false_statement:
            final_label = 'T'
        else:
            final_label = 'F'
    else:
        final_label = 'E'

    print('Set answer: ', answer)
    print('Set final label: ', final_label)
    print('Set false_statement: ', false_statement)
    print('-----------------')
    return final_label


def get_judge_labels(batch: torch.Tensor, kg: KG, llm_question_cache: dict, llm_client=None, llm_answer_cache=None,
                     ask_times=3,
                     *list_args, **kwargs) -> LLMTrainData:
    # negative samples
    golden_false_triples = list()
    false_triple_real_labels = list()
    for h, r, t in batch:
        h, r, t = h.item(), r.item(), t.item()
        fake_h = random.randrange(0, len(kg.entitynames))
        fake_t = random.randrange(0, len(kg.entitynames))
        fake_r = random.randrange(0, len(kg.relationnames))
        choose_which = random.choice([0, 1, 2])
        fake_triples = [(fake_h, r, t), (h, fake_r, t), (h, r, fake_t)]
        fake_triple_labels = [(False, True, True), (False, True, False), (True, True, False)]
        golden_false_triples.append(fake_triples[choose_which])
        false_triple_real_labels.append(copy.deepcopy(fake_triple_labels[choose_which]))

    judge_false_construct_instruct = judge_construct_instruct

    # def judge_false_construct_instruct(question, option_str):
    #     return (
    #             '''### Instruction:\n You are given a false statement. Do you know the statement is false? Answer 'I know the statement is false.' if you know the statement is false. Otherwise, answer 'I don't know the statement is false.' ''' +
    #             # '''If you do not know whether the statement is false, answer "I don't know".''' +
    #             '\n\n' +
    #             "### Input: " + question + '\n\n ### Response: ')

    judge_label_transform_false = partial(judge_label_transform, false_statement=True)

    def judge_label_compare(pred, gold):
        if gold == 'E':
            return False

        return pred == 'T'

    judge_label_negative_compare = judge_label_compare
    # def judge_label_negative_compare(pred, gold):
    #     if gold == 'E':
    #         return False
    #
    #     return pred == 'F'

    true_data = []
    false_data = []

    for i in range(ask_times):
        curr_false_data = get_labels(torch.tensor(golden_false_triples), kg,
                                     llm_answer_cache,
                                     judge_label_transform_false,
                                     llm_client,
                                     judge_false_construct_instruct,
                                     1,
                                     judge_label_negative_compare,
                                     false_data_given=True,
                                     *list_args, **kwargs)
        curr_false_data._set_real_labels(false_triple_real_labels)
        false_data.append(curr_false_data)

    for i in range(ask_times):
        curr_true_data = get_labels(batch, kg,
                                    llm_answer_cache,
                                    judge_label_transform,
                                    llm_client,
                                    judge_construct_instruct,
                                    1,
                                    judge_label_compare,
                                    false_data_given=False,
                                    *list_args, **kwargs)
        curr_true_data._all_true()
        true_data.append(curr_true_data)

    # breakpoint()
    print('true data: ')
    for i in range(ask_times):
        print(true_data[i])
    print('false data: ')
    for i in range(ask_times):
        print(false_data[i])
    print('combined multiple asks:')
    new_true_data = LLMTrainData.majority_vote_judge(true_data)
    new_false_data = LLMTrainData.majority_vote_judge(false_data)

    true_data = new_true_data
    false_data = new_false_data
    print('true data: ', true_data)
    print('false data: ', false_data)
    true_data._merge(false_data)
    print('merged data: ', true_data)
    return true_data


#  Yes, I think the statement is correct. Kevin Garcia is from Liverpool, which is a city in England known for its rich history and culture. A E

def get_labels(batch: torch.Tensor, kg: KG, llm_question_cache: dict, label_transform=get_choice, llm_client=None,
               construct_instruction=default_construct_instruct, option_num=4,
               label_compare=lambda pred, gold: pred == gold, false_data_given=False,
               *list_args, **kwargs) -> LLMTrainData:
    # batch: (batch_size, 3)
    triple_info = BatchTripleInfo(kg)
    if llm_question_cache is None:
        raise ValueError('llm_question_cache cannot be None')
    if llm_client is None:
        llm_client = AskModel()
    questions, answers, options = [], [], []

    for h, r, t in tqdm(batch):
        h, r, t = h.item(), r.item(), t.item()
        option_str, answer, option_ids = kg.provide_tail_options(h, r, t, option_hook=None, option_size=option_num)
        if 'tail' in llm_question_cache[kg.relationnames[r]]:
            question = llm_question_cache[kg.relationnames[r]].format(head=kg.entitynames[h], tail=kg.entitynames[t])
        else:
            question = llm_question_cache[kg.relationnames[r]].format(head=kg.entitynames[h])
        question_with_options = construct_instruction(question, option_str)

        questions.append(question_with_options)
        answers.append(answer)
        options.append(option_ids)
        # print('The Question With Options is: ', question_with_options)
        # print('------------------------------------')

    batch_size = kwargs.get('batch_size', 4)
    labels = {}

    # new_q = []
    # new_a = []
    # old_index = []
    llm_answers = {}
    llm_answer_sequence = []
    # for i in range(len(questions)):
    # new_q.append(questions[i])
    # new_a.append(answers[i])
    # old_index.append(i)
    llm_call_time = 0

    for i in trange(0, len(questions), batch_size, desc='asking questions'):
        q_batch = questions[i:i + batch_size]
        a_batch = questions[i:i + batch_size]
        current_time = time.time()
        llm_answer = llm_client(q_batch)
        llm_call_time += time.time() - current_time
        print('Average call time: ', llm_call_time / (i + 1), ' Avg speed: ', (i + 1) / llm_call_time, 'Model is ', llm_client.model_name)
        # update cache
        for j, qs in enumerate(q_batch):
            llm_answers[qs] = llm_answer[j]
            llm_answer_sequence.append(llm_answer[j])
        for j, answer in enumerate(llm_answer):
            labels[i + j] = label_transform(answer)
            print(answer, a_batch[j], labels[i + j])

    # save cache to cache_path
    # torch.save(llm_answer_cache, cache_path)
    for i, (h, r, t) in enumerate(batch):
        h, r, t = h.item(), r.item(), t.item()
        triple_info.add_triple_info(h, r, t, questions[i], llm_answer_sequence[i], labels[i], not false_data_given)
    # sort values by key
    labels = [labels[i] for i in range(len(labels))]
    train_triples_positive = set()
    train_triples_negative = set()
    for (h, r, t), option_ids, label in zip(batch, options, labels):
        if label_compare(label, 'E'):
            for i in range(len(option_ids)):
                train_triples_negative.add((h, r, option_ids[i]))

        else:
            # train_triples.append((h, r, option_ids[ord(label) - ord('A')]))
            for i in range(len(option_ids)):
                if label_compare(label, chr(ord('A') + i)):
                    train_triples_positive.add((h, r, option_ids[i]))
                else:
                    train_triples_negative.add((h, r, option_ids[i]))
    train_triples_negative = list(train_triples_negative)
    train_triples_positive = list(train_triples_positive)
    return LLMTrainData(torch.tensor(train_triples_positive), torch.tensor(train_triples_negative), batch, options,
                        labels, llm_answers, triple_info=triple_info)

# positive data:  LLMTrainData(total=1000, true=902, e=6, truthful=908, informative=994,truethful_rate=0.9079999923706055, informative_rate=0.9940000176429749,correct_rate=0.9020000100135803, e_rate=0.006000000052154064)
# negative data:  LLMTrainData(total=2000, true=719, e=13, truthful=732, informative=1987,truethful_rate=0.3659999966621399, informative_rate=0.9934999942779541,correct_rate=0.359499990940094, e_rate=0.006500000134110451)
# successfull merge
# merged data:  LLMTrainData(total=3000, true=1621, e=19, truthful=1640, informative=2981,truethful_rate=0.54666668176651, informative_rate=0.9936666488647461,correct_rate=0.5403333306312561, e_rate=0.0063333334401249886)

# positive data:  LLMTrainData(total=1000, true=555, e=443, truthful=998, informative=557,truethful_rate=0.9980000257492065, informative_rate=0.5569999814033508,correct_rate=0.5550000071525574, e_rate=0.4429999887943268)
# negative data:  LLMTrainData(total=3000, true=17, e=1658, truthful=1675, informative=1342,truethful_rate=0.5583333373069763, informative_rate=0.44733333587646484,correct_rate=0.00566666666418314, e_rate=0.5526666641235352)
# successfull merge
# merged data:  LLMTrainData(total=4000, true=574, e=2101, truthful=2674, informative=1899,truethful_rate=0.6685000061988831, informative_rate=0.47475001215934753,correct_rate=0.14350000023841858, e_rate=0.5252500176429749)
