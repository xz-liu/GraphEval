from data_gen.triple_sampler import preprocess_label
from data_gen.triple_sampler import prepare_random_triples
from utils import *

# test_case_name = '3K_triples_2'
test_case_name = str(args.test_case_name)
num_triples = int(args.num_triples)
if __name__ == '__main__':
    kg = get_kg(args)

    triples = prepare_random_triples(kg, num_triples=num_triples)
    print('test_case_name: ', test_case_name)
    preprocess_label(kg, triples, test_case_name=test_case_name,model_name=args.model_name, model_type=args.model_type)

