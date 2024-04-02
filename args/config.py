import os
from os.path import join as pjoin
from .global_args import args

supported_kgs = ['dbpedia-en', 'dbpedia-en-filtered', 'umls']
sentence_transformers_cache_dir = os.environ['SENTENCE_TRANSFORMER_CACHE']

kg_question_paths = {
    'dbpedia-en': pjoin(args.answer_cache_dir, 'dbpedia-en', 'questions.pt'),
    'dbpedia-en-filtered': pjoin(args.answer_cache_dir, 'dbpedia-en', 'questions.pt'),
    'umls': pjoin(args.answer_cache_dir, 'umls', 'questions.pt'),
}

kg_answer_paths = {
    'dbpedia-en': pjoin(args.answer_cache_dir, 'dbpedia-en', 'answers.pt'),
    'dbpedia-en-filtered': pjoin(args.answer_cache_dir, 'dbpedia-en', 'answers.pt'),
    'umls': pjoin(args.answer_cache_dir, 'umls', 'answers.pt'),
}


# TODO remove hf_token

def training_triples_path(model_name, test_case_name='undefined', cache_dir=None, which=None):
    if cache_dir is None:
        cache_dir = args.answer_cache_dir
    if which is None:
        which = args.which
    return os.path.join(cache_dir, 'labels', which, model_name, f'training_triples_{test_case_name}.pt')


def trained_clf_path(model_name, test_case_name='undefined', cache_dir=None, which=None):
    if cache_dir is None:
        cache_dir = args.answer_cache_dir
    if which is None:
        which = args.which
    return os.path.join(cache_dir, 'classifiers', which, model_name, f'trained_clf_{test_case_name}.pt')


def trained_prompt_encoder_path(model_name, test_case_name='undefined', cache_dir=None, which=None):
    if cache_dir is None:
        cache_dir = args.answer_cache_dir
    if which is None:
        which = args.which
    return os.path.join(cache_dir, 'prompt_encoders', which, model_name, f'trained_prompt_encoder_{test_case_name}.pt')


def evaluation_results_path(model_name, test_case_name='undefined', cache_dir=None, which=None, batch_idx=0):
    if cache_dir is None:
        cache_dir = args.answer_cache_dir
    if which is None:
        which = args.which
    return os.path.join(cache_dir, 'evaluation_results', which, model_name,
                        f'evaluation_results_{test_case_name}_batch_{batch_idx}.pt')


def all_evaluation_results(model_name, test_case_name='undefined', cache_dir=None, which=None):
    if cache_dir is None:
        cache_dir = args.answer_cache_dir
    if which is None:
        which = args.which
    result_dir = os.path.join(cache_dir, 'evaluation_results', which, model_name)
    if not os.path.exists(result_dir):
        # create the directory
        os.makedirs(result_dir, exist_ok=True)
        return []

    # get the list of all the files in directory, absolute path
    files = [os.path.join(result_dir, f) for f in os.listdir(result_dir) if
             os.path.isfile(os.path.join(result_dir, f) )]
    # print(files)
    # filter out by test_case_name
    files = [f for f in files if test_case_name in f]
    # print('after filtering', files)
    return files

def analysis_path(model_name, test_case_name='undefined', cache_dir=None, which=None, stage='analysis', format='pt'):
    if cache_dir is None:
        cache_dir = args.answer_cache_dir
    if which is None:
        which = args.which
    return os.path.join(cache_dir, 'analysis_debug', which, model_name, f'analysis_{test_case_name}_{stage}.{format}')


def processed_dataset_path(model_name, test_case_name='undefined', cache_dir=None, which=None):
    if cache_dir is None:
        cache_dir = args.answer_cache_dir
    if which is None:
        which = args.which
    return os.path.join(cache_dir, 'processed_datasets', which, model_name, f'processed_dataset_{test_case_name}.pt')


hf_token = None
