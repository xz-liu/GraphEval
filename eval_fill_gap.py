from evaluator import *

if __name__ == '__main__':
    # for each args

    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg), type(getattr(args, arg))}")

    device_ids = args.device_ids
    if device_ids is not None:
        device_ids = list(map(int, device_ids.split(',')))

    print(f"use device_ids: {device_ids}")

    kg = get_kg(args)
    from evaluator.evaluator import Evaluator

    distributed = 'ddp' if len(device_ids) > 1 else 'none'
    model_name = args.model_name
    model_type = args.model_type
    test_case_name = args.test_case_name
    eval_positive_triples = args.eval_positive_triples.lower() == 'true'
    eval_negative_triples = args.eval_negative_triples.lower() == 'true'
    print(f'eval_positive_triples: {eval_positive_triples}, eval_negative_triples: {eval_negative_triples}')

    ev = Evaluator(model_name, model_type, kg,
                   test_case_name=test_case_name,
                   model_args=dict(torch_dtype=torch.float16, use_flash_attention_2= not args.no_flash_attention),
                   load_model=False, auto_cast=torch.float16
                   )
    ev.fill(kg, kg.triples,
            batch_size=args.eval_batch_size, add_instructions=args.add_instructions,
            device_ids=device_ids, save_each=args.save_each, distributed=distributed,
            eval_positive_triples=eval_positive_triples,
            eval_negative_triples=eval_negative_triples)
