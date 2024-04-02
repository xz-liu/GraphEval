import queue

from trainer.llm_state_loader import *
from peft import PeftModel
from args import *
import time
from .common import *
import torch.utils.data


def dispatch_evaluation_tasks(input_queues, questions, negative_questions, triples, negative_triples,
                              negative_original_triples, batch_size,
                              world_size):
    idx = 0
    assert len(questions) == len(triples)
    assert len(negative_questions) == len(negative_triples)
    for qs, ts, corrupted, ots in zip([questions, negative_questions], [triples, negative_triples], [False, True],
                                      [triples, negative_original_triples]):

        for batch_idx in trange(0, len(qs), batch_size, desc="Dispatching evaluation tasks"):
            batch_end = min(batch_idx + batch_size, len(qs))

            current_results = [
                EvaluationResultUnit(*triple,
                                     *original_triple,
                                     question,
                                     corrupted) for
                triple, original_triple, question in
                zip(ts[batch_idx:batch_end], ots[batch_idx:batch_end], qs[batch_idx:batch_end])]

            input_queues[idx % world_size].put(current_results)
            idx += 1
            time.sleep(0.001)

    for input_queue in input_queues:
        input_queue.put('done')


import traceback


def handling_evaluation_results(output_queues, len_triples, result_path_args, save_each=1000, input_dataset=None,
                                ):
    ib = 0
    result_batch = len(all_evaluation_results(**result_path_args))
    formatted_results = []
    # TODO mask broken, only use input_only
    idx_mask = None
    len_recv = 0
    pbar = tqdm(total=len_triples, desc='Handling evaluation results')
    while len_recv < len_triples:
        current_results = None
        while current_results is None:

            for idx, q in enumerate(output_queues):
                try:
                    current_results = q.get_nowait()
                    if current_results is not None:
                        # print('recieved from device', idx, 'units:', len(current_results))
                        break
                # empty queue exception
                except queue.Empty as e:
                    pass
            if current_results is not None:
                break
            time.sleep(0.1)
            # print('units_recieved:', units_recieved, 'current queue sizes:', [q.qsize() for q in output_queues])

        assert isinstance(current_results, BatchEvaluationOutput)
        for i in range(len(current_results)):
            idx = current_results.get_idx(i)
            result = current_results[i]
            formatted_results.append(EvaluationResultUnit.from_dict({**input_dataset.get_whole_input(idx), **result}))

        if ib % save_each == 0:
            verbose_evaluation_progress(formatted_results, idx_mask)
            result_to_save = formatted_results
            result_path = evaluation_results_path(**result_path_args,batch_idx=result_batch)
            save_to(result_path, result_to_save)
            result_batch += 1
            formatted_results = []
        ib += 1
        pbar.update(len(current_results))
        len_recv += len(current_results)
    pbar.close()
    verbose_evaluation_progress(formatted_results, idx_mask)
    save_to(evaluation_results_path(**result_path_args, batch_idx=result_batch), formatted_results)


def multi_device_serving(rank, model: EvaluationModel, tokenizer, input_queue: List, output_queue: List):
    print(f"Device {rank} serving...")
    device = torch.device(f'cuda:{rank}')
    input_queue = input_queue[rank]
    output_queue = output_queue[rank]
    model.to(device)
    model.eval()
    total_units = 0
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        with torch.no_grad():
            while True:
                batch = None
                while batch is None:
                    try:
                        batch: List[EvaluationResultUnit] = input_queue.get_nowait()
                    except:
                        time.sleep(0.1)

                if batch == 'done':
                    print(f"Device {rank} serving done, {total_units} units served.")
                    break
                batch_sentences = [unit.prompt for unit in batch]
                torch.cuda.synchronize(device)
                before = time.perf_counter()
                inputs = tokenizer(batch_sentences, return_tensors="pt", max_length=256, padding=True, truncation=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                true_false = torch.tensor([unit.corrupted for unit in batch]).to(device)
                true_false = torch.bitwise_not(true_false).long()
                result = model(**inputs, true_false=true_false)
                torch.cuda.synchronize(device)
                after = time.perf_counter()
                inference_time = (after - before) / float(len(batch))
                new_units = []
                for i, unit in enumerate(batch):
                    new_units.append(unit.clone())
                    new_units[-1].set_evaluation_results(result[i], torch.argmax(result[i]), inference_time)
                output_queue.put(new_units)
                total_units += len(new_units)
