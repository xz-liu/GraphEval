from transformers import AutoModel, AutoTokenizer, AutoConfig, LlamaTokenizer, LlamaModel, AutoModel, \
    LlamaForCausalLM, LlamaConfig, AutoModelForCausalLM
from trainer.llm_state_loader import *
from peft import PeftModel
from args import *
import time

import torch.multiprocessing as mp
import torch.distributed as dist
import torch.utils.data

from .common import EvaluationResultUnit, EvaluationModel, QueriesDataset, verbose_evaluation_progress, \
    BatchEvaluationInput, BatchEvaluationOutput
from .mp_helper import dispatch_evaluation_tasks, handling_evaluation_results, multi_device_serving
from .dist_helper import prepare_dataloader, generate_evaluation_tasks, ddp_process

import random
from collections import defaultdict


# from .common import EvaluationResultUnit, EvaluationModel, QueriesDataset


class Evaluator:
    def __init__(self, model_name, model_type, kg, test_case_name, model_args: Dict = None, load_model=False,
                 auto_cast=None):
        if model_args is None:
            model_args = {}
        self.model_path_dict = dict(
            model_name=model_name,
            model_type=model_type,
            peft_path=trained_prompt_encoder_path(model_name, test_case_name, which=kg.which),
            classifier_path=trained_clf_path(model_name, test_case_name, which=kg.which),
            model_args=model_args
        )
        self.tokenizer = None
        # self.model
        self.load_model = load_model
        self.model_name = model_name
        self.kg = kg
        self.model_type = model_type
        self.test_case_name = test_case_name
        # load answer template
        self.answer_template = load_from(kg_answer_paths[kg.which])
        if load_model:
            self._load_model()
        self.auto_cast = auto_cast

    def _load_model(self):
        # model_name = self.model_path_dict['model_name']
        # model_args = self.model_path_dict['model_args']
        # peft_path = self.model_path_dict['peft_path']
        # classifier_path = self.model_path_dict['classifier_path']
        # model = AutoModelForCausalLM.from_pretrained(model_name, **model_args)
        # classifier = load_from(classifier_path)
        # if os.path.exists(peft_path):
        #     model = PeftModel.from_pretrained(model, peft_path)
        # # self.model.to(device)/
        self.model = EvaluationModel.from_path_dict(self.model_path_dict)

        self.tokenizer = AutoTokenizer.from_pretrained(small_lm_of(self.model_name))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"

    def negative_sampling(self, triples, samples_per_triple=5):
        kg = self.kg
        if samples_per_triple <= 0:
            return [], []

        fake_triples, their_original = [], []
        for h, r, t in tqdm(triples,
                            desc=f"Negative sampling for {len(triples)} triples, samples_per_triple={samples_per_triple}"):
            h, r, t = int(h), int(r), int(t)
            for _ in range(samples_per_triple):
                fake_h = random.randrange(0, len(kg.entitynames))
                fake_t = random.randrange(0, len(kg.entitynames))
                fake_r = random.randrange(0, len(kg.relationnames))
                choose_which = random.choice([0, 1, 2])
                fake_triple_choices = [(fake_h, r, t), (h, fake_r, t), (h, r, fake_t)]
                fake_triples.append(fake_triple_choices[choose_which])
                their_original.append((h, r, t))
        return fake_triples, their_original

    def _filling_gap_triples(self, triples):
        # print()
        all_results = all_evaluation_results(self.model_name, self.test_case_name, which=self.kg.which)
        print(f"Found {len(all_results)} results")
        positive_counts = {(int(triple[0]), int(triple[1]), int(triple[2])): 0 for triple in triples}
        negative_counts = {(int(triple[0]), int(triple[1]), int(triple[2])): 0 for triple in triples}
        max_positive = 0
        max_negative = 0
        pbar = tqdm(total=len(all_results))
        for result_path in all_results:
            result_pack = load_from(result_path, verbose=False)
            for result in result_pack:
                if result.corrupted:
                    tr = result.original_triple()
                    negative_counts[tr] += 1
                    max_negative = max(max_negative, negative_counts[tr])
                else:
                    tr = result.original_triple()
                    positive_counts[tr] += 1
                    max_positive = max(max_positive, positive_counts[tr])
            pbar.set_description('counting results...')
            pbar.set_postfix(
                pm=max_positive,
                nm=max_negative,
                f=result_path
            )
            pbar.update(1)

        pbar.close()
        print('max_positive:', max_positive)
        print('max_negative:', max_negative)

        positive_filling_triples = defaultdict(list)
        negative_filling_triples = defaultdict(list)

        for tr, cnt in tqdm(positive_counts.items(), desc='count filling for positive triples'):
            if cnt < max_positive:
                positive_filling_triples[max_positive - cnt].append(tr)

        for tr, cnt in tqdm(negative_counts.items(), desc='count filling for negative triples'):
            if cnt < max_negative:
                negative_filling_triples[max_negative - cnt].append(tr)

        print('Count filling done')
        for k, v in positive_filling_triples.items():
            print(f'Filling {k} positive triples for {len(v)} triples')

        for k, v in negative_filling_triples.items():
            print(f'Filling {k} negative triples for {len(v)} triples')

        fake_triples = []
        their_original = []
        for k, v in negative_filling_triples.items():
            curr_fake_triples, curr_their_original = self.negative_sampling(v, k)
            fake_triples.extend(curr_fake_triples)
            their_original.extend(curr_their_original)

        return positive_filling_triples[1], fake_triples, their_original

    def _get_questions(self, triples, add_instructions, kg_questions, kg_answers,
                       desc='constructing questions for positive triples'):
        questions = []
        for triple in tqdm(triples, desc=desc):
            triple = [int(triple[0]), int(triple[1]), int(triple[2])]
            question = \
                construct_questions(self.kg, triple, kg_questions, kg_answers, ty=self.model_type,
                                    add_instructions=add_instructions )[0]
            questions.append(question)
        return questions

    def _run(self, kg, triples, fake_triples, their_original,
             batch_size=8, add_instructions=True, device_ids=(0, 1, 2, 3),
             save_each=3000, eval_positive_triples=True, eval_negative_triples=True,
             distributed='ddp'):

        kg_questions = load_from(kg_question_paths[kg.which])
        kg_answers = load_from(kg_answer_paths[kg.which])
        questions = self._get_questions(triples, add_instructions, kg_questions, kg_answers,
                                        desc='constructing questions for positive triples')

        negative_questions = self._get_questions(fake_triples, add_instructions, kg_questions, kg_answers,
                                                 desc='constructing questions for negative triples')

        assert distributed in ['lightning', 'ddp', 'mp', 'none', 'huggingface'], "distributed mode not supported"
        getattr(self, f"_{distributed}_distributed_run")(questions, negative_questions, triples,
                                                         fake_triples, their_original, batch_size,
                                                         save_each, device_ids, eval_positive_triples,
                                                         eval_negative_triples)

    @torch.no_grad()
    def fill(self, kg, triples, batch_size=8, add_instructions=False, device_ids=(0, 1, 2, 3),save_each=3000,distributed='none',
             eval_positive_triples=True, eval_negative_triples=True):
        positive_filling_triples, fake_triples, their_original = self._filling_gap_triples(triples)
        self._run(kg, positive_filling_triples, fake_triples, their_original, batch_size, add_instructions, device_ids,
                  save_each, eval_positive_triples, eval_negative_triples, distributed=distributed)

    @torch.no_grad()
    def __call__(self, kg, triples,
                 batch_size=8, add_instructions=False, device_ids=(0, 1, 2, 3), save_each=3000,
                 negative_sampling_ratio=5, eval_positive_triples=True, eval_negative_triples=True,
                 distributed='ddp'):

        fake_triples, their_original = self.negative_sampling(triples, negative_sampling_ratio)
        self._run(kg, triples, fake_triples, their_original, batch_size, add_instructions, device_ids, save_each,
                  eval_positive_triples, eval_negative_triples, distributed)

    def _generation_run(self, questions, negative_questions, triples, fake_triples, their_original, batch_size,
                        save_each, device_ids, eval_positive_triples, eval_negative_triples):

        from transformers import pipeline
        generator = pipeline('text-generation', model=self.model.peft_model, tokenizer=self.tokenizer,
                             device=torch.device('cuda:{}'.format(device_ids[0])), torch_dtype=torch.bfloat16)

        def data_generator():
            for question in questions:
                yield question

        for i, resp in tqdm(enumerate(generator(data_generator(),
                                                do_sample=True,
                                                num_return_sequences=1,
                                                eos_token_id=self.tokenizer.eos_token_id,
                                                max_new_tokens=50,  # max lenght of output, default=4096
                                                return_full_text=False,  # to not repeat the question, set to False
                                                top_k=10,  # default=10
                                                top_p=0.9,  # default=0.9
                                                temperature=0.6,  # default=0.
                                                batch_size=batch_size,
                                                ))):
            # all_results.append(map(lambda x: x['generated_text'], resp))
            print(resp)

    def _none_distributed_run(self, _questions, _negative_questions, _triples, _fake_triples, _their_original,
                              batch_size,
                              save_each, device_ids, eval_positive_triples, eval_negative_triples):
        # TODO support negative sampling
        ib = 0
        device = torch.device(f'cuda:{device_ids[0]}')
        if not self.load_model:
            self._load_model()

        self.model.to(device)
        formatted_results = []
        with torch.autocast(device_type='cuda', dtype=self.auto_cast) if self.auto_cast is not None else nullcontext():
            for questions, triples, original_triples, do_eval, corrupted in zip([_questions, _negative_questions],
                                                                                [_triples, _fake_triples],
                                                                                [_triples, _their_original],
                                                                                [eval_positive_triples,
                                                                                 eval_negative_triples],
                                                                                [False, True]):
                if not do_eval:
                    continue
                for batch_idx in trange(0, len(questions), batch_size):
                    batch_end = min(batch_idx + batch_size, len(questions))
                    batch = questions[batch_idx:batch_end]
                    current_results = [
                        EvaluationResultUnit(triple[0], triple[1], triple[2], original_triple[0], original_triple[1],
                                             original_triple[2],
                                             question, corrupted) for
                        triple, question, original_triple in
                        zip(triples[batch_idx:batch_end], batch, original_triples[batch_idx:batch_end])]
                    curr_time = time.perf_counter()
                    inputs = self.tokenizer(batch, return_tensors="pt", max_length=256, padding=True,
                                            truncation=True)
                    inputs.to(device)
                    result = self.model(inputs['input_ids'], inputs['attention_mask'],
                                        torch.tensor([int(not corrupted)] * len(batch)).to(device))

                    inference_time = time.perf_counter() - curr_time
                    inference_time /= len(batch) + 1e-6
                    pred = torch.argmax(result, dim=1).cpu()
                    for i, unit in enumerate(current_results):
                        unit.set_evaluation_results(result[i], pred[i], inference_time)
                    formatted_results.extend(current_results)
                    if save_each > 0 and ib % save_each == 0:
                        print(f"batch {batch_idx} done")
                        verbose_evaluation_progress(formatted_results)
                        save_to(evaluation_results_path(
                            self.model_name, self.test_case_name, which=self.kg.which,
                            batch_idx=len(all_evaluation_results(
                                self.model_name, self.test_case_name, which=self.kg.which
                            ))
                        ), formatted_results)
                        formatted_results = []
                    ib += 1
            verbose_evaluation_progress(formatted_results)
            save_to(evaluation_results_path(
                self.model_name, self.test_case_name, which=self.kg.which, batch_idx=len(all_evaluation_results(
                    self.model_name, self.test_case_name, which=self.kg.which
                ))
            ), formatted_results)

    def _ddp_distributed_run(self, questions, negative_questions, triples, fake_triples, their_original, batch_size,
                             save_each, device_ids, eval_positive_triples, eval_negative_triples):
        # TODO share memory
        mp.set_sharing_strategy('file_system')
        manager = mp.Manager()
        output_queue = manager.Queue()

        # processed_path = processed_dataset_path(self.model_name, self.test_case_name, which=self.kg.which)
        queries = generate_evaluation_tasks(questions, negative_questions, triples, fake_triples,
                                            their_original, batch_size, eval_positive_triples, eval_negative_triples)
        queries = BatchEvaluationInput.from_list(queries)
        queries.share_memory()
        # create model
        model = EvaluationModel.from_path_dict(self.model_path_dict)
        model.share_memory()
        print('Creating handler...')
        handler = mp.Process(target=handling_evaluation_results,
                             args=([output_queue], len(queries), dict(
                                 model_name=self.model_name,
                                 test_case_name=self.test_case_name,
                                 which=self.kg.which,
                             ), save_each, queries))
        handler.start()
        print("Starting spawn ddp process...")
        mp.spawn(ddp_process,
                 args=(len(device_ids), model, self.model_name, queries, batch_size, output_queue, self.auto_cast),
                 nprocs=len(device_ids))
        handler.join()
