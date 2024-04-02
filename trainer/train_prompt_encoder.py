from peft import PromptEncoder, PromptEncoderConfig, get_peft_model

import data_gen
from args import *
from utils import *
from tqdm import tqdm, trange
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

from data_gen.data_generator import TripleInfo
from datasets import Dataset
from functools import partial

from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader


def preprocess_function(examples, tokenizer):
    return tokenizer(examples['text'], return_tensors='pt', truncation=True, padding=True, max_length=256)


def train_prompt_encoder(kg, test_case_name, model_name, model_type):
    from args import kg_question_paths, kg_answer_paths
    save_dir = trained_prompt_encoder_path(model_name, test_case_name=test_case_name,which=args.which)
    print('model will be saved at: ', save_dir)
    llm_question_cache = load_from(kg_question_paths[args.which])
    llm_answer_cache = load_from(kg_answer_paths[args.which])

    data = data_gen.get_training_triples(kg, test_case_name=test_case_name, model_name=model_name, model_type=model_type)
    data._preprocess()
    # constructing QA pairs
    qa_pairs = []
    for ti in tqdm(data.triple_info.triple_info.values()):
        ti: TripleInfo = ti[0]

        h, r, t = ti.h, ti.r, ti.t
        ans = ti.llm_response
        question = construct_questions(kg, (h, r, t), llm_question_cache, llm_answer_cache,ty= model_type, add_instructions=False)[0]
        qa_pairs.append(' '.join([question, ans]))

    dataset = Dataset.from_dict({'text': qa_pairs})
    # train test split
    dataset = dataset.train_test_split(test_size=0.1)
    model_name= small_lm_of(model_name)
    print('use',model_name, 'to train peft')
    tokenizer = AutoTokenizer.from_pretrained(small_lm_of(model_name))
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    dataset = dataset.map(partial(preprocess_function, tokenizer=tokenizer), batched=True,
                          remove_columns=dataset['train'].column_names)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # PromptEncoder
    transformer_model = AutoModelForCausalLM.from_pretrained(model_name)


    config = PromptEncoderConfig(
        peft_type="P_TUNING",
        task_type="CAUSAL_LM",
        num_virtual_tokens=20,
        token_dim=transformer_model.config.hidden_size,
        num_transformer_submodules=1,
        num_attention_heads=12,
        num_layers=12,
        encoder_reparameterization_type="MLP",
        encoder_hidden_size=4096,
    )

    transformer_model = get_peft_model(transformer_model, config)
    transformer_model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=save_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        push_to_hub=False,
        num_train_epochs=5,
        save_strategy="epoch",
        per_device_train_batch_size=args.peft_batch_size,
        per_device_eval_batch_size=args.peft_batch_size,
        dataloader_num_workers=4,
        dataloader_prefetch_factor=2,

    )

    trainer = Trainer(
        model=transformer_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
    )

    trainer.train()
    import math
    transformer_model.save_pretrained(save_dir)
    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")


if __name__ == '__main__':
    # p_tuning()
    pass
