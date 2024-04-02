import argparse
import os

parser = argparse.ArgumentParser()

# cache path
parser.add_argument('--answer_cache_dir', type=str, default='cache/', help='answer cache dir')
parser.add_argument('--cache_path', type=str, default='cache/', help='cache path')
parser.add_argument('--hidden_states_cache_path', type=str, default='cache/hidden_states/', help='cache path')

# model args
parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-chat-hf", help='model name')
parser.add_argument('--model_type', type=str, default="llama", help='model type')
parser.add_argument('--test_case_name', type=str, default='standard_2k_judge')

# KG args
parser.add_argument('--which', type=str, default='dbpedia-en-filtered', help='which KG to use')
parser.add_argument('--kg_path', type=str,
                    default='../kg/dbpedia/mappingbased-objects_lang=en.ttl.bzip2.out',
                    help='kg path')
# embedding args
parser.add_argument('--embedding_dir', type=str, default='./cache/kgs/', help='embedding dir')
parser.add_argument('--device', type=str, default='cuda', help='device')

# training args
parser.add_argument('--peft_batch_size', type=int, default=4, help='peft batch size')

# eval args
parser.add_argument('--eval_batch_size', type=int, default=64, help='batch size for evaluation')
parser.add_argument('--add_instructions', action='store_true', help='add instructions')
parser.add_argument('--use_generation', type=bool, default=False, help='use generation')
parser.add_argument('--device_ids', type=str, default="0,1,2,3", help='device ids')
parser.add_argument('--save_each', type=int, default=3000, help='save each')
parser.add_argument('--negative_sampling_ratio', type=int, default=1, help='negative sampling ratio')
parser.add_argument('--eval_positive_triples', type=str, default='True', help='evaluate positive triples')
parser.add_argument('--eval_negative_triples', type=str, default='True', help='evaluate negative triples')

# for collect.py
parser.add_argument('--num_triples', type=int, default=2000)

# ablation study
parser.add_argument('--no_small_lm', action='store_true', help='use small lm')
parser.add_argument('--no_peft', action='store_true', help='use peft')
parser.add_argument('--no_save', action='store_true', help='save results')
parser.add_argument('--substitute_model', type=str, default='None', help='substitute model')
parser.add_argument('--eval_transformer_logits', action='store_true', help='eval transformer logits')
parser.add_argument('--no_flash_attention', action='store_true', help='no flash attention')
parser.add_argument('--debug_total_eval', default=-1, type=int, help='debug total eval')
parser.add_argument('--debug_parameter', type=int, default=0, help='debug parameter')

args = parser.parse_args()

# supported KGs
