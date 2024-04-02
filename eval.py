from evaluator import *

if __name__ == '__main__':
    # for each args
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg), type(getattr(args, arg))}")

    # show current folder
    print(f"current folder: {os.getcwd()}")


    device_ids = args.device_ids
    if device_ids is not None:
        device_ids = list(map(int, device_ids.split(',')))

    print(f"use device_ids: {device_ids}")

    print('cuda available:', torch.cuda.is_available())
    print('cuda device count:', torch.cuda.device_count())
    print('cuda device:', torch.cuda.current_device())
    for i in range(torch.cuda.device_count()):
        print(f'cuda device {i}: {torch.cuda.get_device_name(i)}')
    # 1.8204214008262505 1.2816117926628308
    kg = get_kg(args)
    from evaluator.evaluator import Evaluator

    model_name = args.model_name
    model_type = args.model_type
    test_case_name = args.test_case_name

    if args.debug_total_eval > 0:
        kg.triples = kg.triples[:args.debug_total_eval]

    eval_positive_triples = args.eval_positive_triples.lower() == 'true'
    eval_negative_triples = args.eval_negative_triples.lower() == 'true'
    print(f'eval_positive_triples: {eval_positive_triples}, eval_negative_triples: {eval_negative_triples}')
    # ev = Evaluator(model_name, model_type, kg,
    #                test_case_name=test_case_name,
    #                model_args=dict(torch_dtype=torch.bfloat16, use_flash_attention_2= not args.no_flash_attention)
    #                )
    ev = Evaluator(model_name, model_type, kg,
                   test_case_name=test_case_name,
                   model_args=dict(torch_dtype=torch.float16, use_flash_attention_2= not args.no_flash_attention)
                   ,auto_cast=torch.float16,load_model=False)
    distributed_mode= 'ddp' if len(device_ids) > 1 else 'none'
    ev(kg, kg.triples,
       batch_size=args.eval_batch_size, add_instructions=args.add_instructions,
       device_ids=device_ids, save_each=args.save_each, negative_sampling_ratio=args.negative_sampling_ratio,
       eval_positive_triples=eval_positive_triples, eval_negative_triples=eval_negative_triples,
       distributed=distributed_mode
       )