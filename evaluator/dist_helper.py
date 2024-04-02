from trainer.llm_state_loader import *
from peft import PeftModel
from args import *
import time

import torch.utils.data

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from .common import *


def prepare_dataloader(dataset: QueriesDataset, batch_size: int, rank: int, world_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset, num_replicas=world_size, rank=rank),
        # num_workers=8,
    )


def generate_evaluation_tasks(questions, negative_questions, triples, negative_triples,
                              negative_original_triples, batch_size, eval_positive_triples, eval_negative_triples) \
        -> List[EvaluationResultUnit]:
    # idx = 0
    assert len(questions) == len(triples)
    assert len(negative_questions) == len(negative_triples)
    queries = []
    for qs, ts, corrupted, ots, do_eval in zip([questions, negative_questions], [triples, negative_triples],
                                               [False, True],
                                               [triples, negative_original_triples],
                                               [eval_positive_triples, eval_negative_triples]):
        if not do_eval:
            print('Skip generating evaluation tasks for', not corrupted, 'triples')
            continue
        for batch_idx in trange(0, len(qs), batch_size, desc="Dispatching evaluation tasks"):
            batch_end = min(batch_idx + batch_size, len(qs))

            current_results = [
                EvaluationResultUnit(*triple,
                                     *original_triple,
                                     question,
                                     corrupted) for
                triple, original_triple, question in
                zip(ts[batch_idx:batch_end], ots[batch_idx:batch_end], qs[batch_idx:batch_end])]

            queries.extend(current_results)
    return queries


@torch.no_grad()
def ddp_process(rank, world_size, model, model_name, dataset: BatchEvaluationInput, batch_size, output_queue,
                auto_cast=None, handler=None):
    print(f"Device {rank} setting up DDP...")
    ddp_setup(rank, world_size)
    print(f"Device {rank} setting up model...")
    tokenizer = AutoTokenizer.from_pretrained(small_lm_of(model_name))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    model.to(rank)
    model = DDP(model, device_ids=[rank])
    device = torch.device(f'cuda:{rank}')
    print(f"Device {rank} loading dataset...")
    data_loader = prepare_dataloader(dataset, batch_size, rank, world_size)
    data_loader.sampler.set_epoch(0)
    with torch.autocast(device_type='cuda', dtype=auto_cast) if auto_cast is not None else nullcontext():
        for batch in data_loader:
            batch_sentences = batch['prompt']
            # torch.cuda.synchronize(device)
            before = time.perf_counter()
            inputs = tokenizer(batch_sentences, return_tensors="pt", max_length=256, padding=True,
                               truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            true_false = batch['corrupted'].to(device)
            true_false = torch.bitwise_not(true_false).long()
            result = model(**inputs, true_false=true_false)
            # torch.cuda.synchronize(device)
            after = time.perf_counter()
            inference_time = (after - before) / float(len(batch['prompt']))
            output = BatchEvaluationOutput(
                batch['idx'].cpu(), result.cpu(), result.argmax(dim=1).cpu(), inference_time
            )
            output_queue.put(output)
    destroy_process_group()


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
