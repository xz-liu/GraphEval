# import collect
import data_gen
from utils import *
from trainer import FineTuneModel
from transformers import AdamW
import torch
import torch.nn.functional as F
import time
from tqdm import tqdm, trange
from data_gen import LLMTrainData
from args import *

from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score


def save_trained_clf(model, model_name, test_case_name='undefined', cache_dir=None, which=None):
    save_path = trained_clf_path(model_name, test_case_name, cache_dir, which)
    model.save_clf(save_path)


def load_trained_clf(model_name, test_case_name='undefined', cache_dir=None, which=None):
    save_path = trained_clf_path(model_name, test_case_name, cache_dir, which)
    return load_from(save_path)


def train_clf_impl(kg, triples, targets, device='cuda', test_samples=-1, n_classes=2,
                   model_name=None, test_case_name='undefined', llm_question_cache=None,
                   llm_answer_cache=None,
                   max_seq_len=32, batch_size=32, data: LLMTrainData = None, use_peft=True, model_type=None):
    peft_path = trained_prompt_encoder_path(model_name,
                                            test_case_name=test_case_name) if use_peft else None
    model_name = small_lm_of(model_name)
    print('use', model_name, 'to train classifier')
    # TODO fix here!!! Important
    add_instructions = not use_peft
    model = FineTuneModel(model_name, -1, max_seq_len, -1, n_classes=n_classes,
                          which_classifier='feedforward',
                          use_cache=True, cache_type='dict', llm_infer='forward', keep_seq_len=1, keep_layers=-1,
                          keep_seq=-1, tf_dim=1, peft_path=peft_path, )

    # model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    from sklearn.model_selection import train_test_split
    triples_train, triples_test, target_train, target_test = train_test_split(triples, targets, test_size=0.3,
                                                                              random_state=42)
    # triples_train, triples_test, target_train, target_test = train_samples
    print('target train', torch.sum(target_train == 1), torch.sum(target_train == 0))
    print('target test', torch.sum(target_test == 1), torch.sum(target_test == 0))
    triples_val, target_val = None, None

    randperm = torch.randperm(len(triples_test))
    triples_test = triples_test[randperm]
    target_test = target_test[randperm]
    # target_train, target_test = targets_train.float(), target_test.float()
    print('constructing questions...')
    if test_samples < 0:
        test_samples = len(triples_test)
    sentences_test = [construct_questions(kg, triple, llm_question_cache, llm_answer_cache, ty=model_type,
                                          add_instructions=add_instructions)[0] for triple
                      in
                      tqdm(triples_test[:test_samples])]
    tf_test = data.get_real_labels(triples_test[:test_samples])
    print('true false test label shape', tf_test.shape)

    target_test = target_test[:test_samples]
    # scaler = torch.cuda.amp.GradScaler()
    for epoch in range(100):
        model.train()
        train_loss = 0
        randperm_train = torch.randperm(len(triples_train))
        triples_train = triples_train[randperm_train]
        target_train = target_train[randperm_train]
        sentences_train = [construct_questions(kg, triple, llm_question_cache, llm_answer_cache, ty=model_type,
                                               add_instructions=add_instructions)[0] for
                           triple in
                           tqdm(triples_train)]
        tf_train = data.get_real_labels(triples_train)
        # pos_train = [get_positions(kg, triple, llm_question_cache, llm_answer_cache) for triple in
        #              tqdm(triples_train)]
        print('true false train label shape', tf_train.shape)
        for i in trange(0, len(sentences_train), batch_size, desc=''):
            batch = sentences_train[i:min(i + batch_size, len(sentences_train))]
            labels = target_train[i:min(i + batch_size, len(sentences_train))].to(device)

            # with torch.cuda.amp.autocast():
            outputs = model(batch, labels=labels, true_false=tf_train[i:min(i + batch_size, len(sentences_train))])

            loss = outputs.loss
            loss.backward()
            # scaler.scale(loss).backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()
            # print('epoch {}, train loss {}'.format(epoch, loss))
        print('epoch', epoch, 'train loss', train_loss)
        # scheduler.step()
        if (epoch + 1) % 1 == 0:
            model.eval()
            with torch.no_grad():
                evaluate_on_dataset(sentences_train, target_train, model, device, epoch,
                                    batch_size=batch_size, max_seq_len=max_seq_len, train=True, n_classes=n_classes,
                                    true_false=tf_train)
                evaluate_on_dataset(sentences_test, target_test, model, device, epoch, batch_size=batch_size,
                                    max_seq_len=max_seq_len, train=False, n_classes=n_classes, triples_val=triples_val,
                                    target_val=target_val, true_false=tf_test)

    return model


def evaluate_on_dataset(sentences_test, target_test, model, device, epoch, batch_size=4, max_seq_len=32,
                        train=False, n_classes=2, triples_val=None, target_val=None, autocast_dtype=None,
                        get_logits=False, **kwargs):
    # test_correct = 0
    # total_test = 0
    # tp, fp, tn, fn = 0, 0, 0, 0
    torch.cuda.synchronize()
    time_now = time.perf_counter()

    total_logits = []

    total_transformer_logits = []
    if epoch == 0 and train == False and args.eval_transformer_logits:
        print('evaluate transformer logits')
        eval_transformer_logits = True
    else:
        eval_transformer_logits = False

    with torch.autocast(device_type='cuda', dtype=autocast_dtype) if autocast_dtype is not None else nullcontext():
        for i in trange(0, len(sentences_test), batch_size):
            batch = sentences_test[i:min(i + batch_size, len(sentences_test))]
            labels = target_test[i:min(i + batch_size, len(sentences_test))].to(device)
            curr_args = {}
            for k, v in kwargs.items():
                curr_args[k] = v[i:min(i + batch_size, len(sentences_test))]
            outputs = model(batch, **curr_args)
            logits = outputs.logits
            total_logits.append(logits.detach().cpu())
            if eval_transformer_logits:
                tlogits = model.transformer_logits(batch, **curr_args)
                total_transformer_logits.append(tlogits.detach().cpu())

    torch.cuda.synchronize()
    total_time = time.perf_counter() - time_now
    total_token = len(sentences_test) * 64
    print('total time', total_time, 'total token', total_token, 'speed', total_token / total_time)

    total_logits = torch.cat(total_logits, dim=0)

    total_logits = total_logits.cpu()

    preds1 = torch.argmax(total_logits, dim=1)
    preds1 = preds1.cpu()
    all_preds = {"clf_logits": preds1}
    if eval_transformer_logits:
        t_logits = torch.cat(total_transformer_logits, dim=0)
        t_logits = t_logits.cpu()
        all_preds["transformer_logits"] = t_logits

    for name, preds in all_preds.items():  # ,preds2]:
        print("Evaluation results for", name)
        # binary classification or multi-class classification
        average = 'binary' if n_classes == 2 else 'macro'

        train = 'train' if train else 'test'
        print('epoch', epoch, train)
        if n_classes == 2:
            print('f1', f1_score(target_test, preds, average=average))
            print('precision', precision_score(target_test, preds, average=average))
            print('recall', recall_score(target_test, preds, average=average))
            print('confusion matrix', confusion_matrix(target_test, preds))
        elif n_classes == 3:
            # 3        print('classification report',
            #               classification_report(target_test, preds, target_names=['correct', 'wrong', 'dont_know']))
            print('confusion matrix', confusion_matrix(target_test, preds))
            print('f1', f1_score(target_test, preds, average='weighted'))
            print('precision', precision_score(target_test, preds, average='weighted'))
            print('recall', recall_score(target_test, preds, average='weighted'))

    if get_logits:
        return total_logits, preds


def train_classifier(kg, model_name, model_type, test_case_name, use_peft=True):
    from args import kg_question_paths, kg_answer_paths

    data = data_gen.get_training_triples(kg, test_case_name=test_case_name, model_name=model_name,
                                         model_type=model_type)
    data._preprocess()
    batch_labels = data.multi_class_labels
    batch = data.batch
    print(data)
    llm_question_cache = load_from(kg_question_paths[args.which])
    llm_answer_cache = load_from(kg_answer_paths[args.which])
    triples, labels = batch, torch.tensor(batch_labels)

    # breakpoint()
    model = train_clf_impl(kg, triples, torch.tensor(labels), test_samples=-1, n_classes=3,
                           llm_question_cache=llm_question_cache,
                           llm_answer_cache=llm_answer_cache, max_seq_len=256, batch_size=4, data=data,
                           model_name=model_name, test_case_name=test_case_name, use_peft=use_peft,
                           model_type=model_type)

    save_trained_clf(model, model_name, test_case_name=test_case_name)


@torch.no_grad()
def estimate_on_another(kg, model_name, model_type, test_case1, test_case2, device='cuda', batch_size=4,
                        max_seq_len=256, n_classes=3, use_peft=True):
    # TODO not complete
    from data_gen import LLMTrainData
    test_case1_data = load_from(training_triples_path(model_name, test_case1, which=kg.which))
    test_case2_data: LLMTrainData = load_from(training_triples_path(model_name, test_case2, which=kg.which))
    llm_question_cache = load_from(kg_question_paths[kg.which])
    llm_answer_cache = load_from(kg_answer_paths[kg.which])
    from trainer.model import FineTuneModel
    evaluation_model = FineTuneModel(model_name, 32, max_seq_len, 4096, n_classes=n_classes,
                                     which_classifier='feedforward',
                                     use_cache=True, cache_type='dict', llm_infer='forward', keep_seq_len=1,
                                     keep_layers=-1,
                                     keep_seq=-1, tf_dim=1,
                                     peft_path=trained_prompt_encoder_path(model_name,
                                                                           test_case_name=test_case1) if use_peft else None,
                                     trained_clf=trained_clf_path(model_name, test_case1, which=kg.which))

    evaluation_model.to(device)
    use_data1 = -1
    use_data2 = -1

    data1_triples = test_case1_data.batch  # [:use_data1]
    data1_targets = torch.tensor(test_case1_data.multi_class_labels)  # [:use_data1]
    data1_tf = test_case1_data.get_real_labels(data1_triples)  # [:use_data1]
    data1_sentences = [construct_questions(kg, triple, llm_question_cache, llm_answer_cache, ty=model_type)[0] for
                       triple in
                       tqdm(data1_triples)]  # [:use_data1]

    data1_logits, _ = evaluate_on_dataset(data1_sentences, torch.tensor(data1_targets),
                                          evaluation_model,
                                          device, 0, batch_size=batch_size,
                                          max_seq_len=max_seq_len, train=False, n_classes=n_classes,
                                          triples_val=None,
                                          target_val=None, true_false=data1_tf, autocast_dtype=torch.float16,
                                          get_logits=True)
    # one hot encoding
    data1_llm_logits = torch.tensor(data1_targets)
    data1_llm_logits = F.one_hot(data1_llm_logits, n_classes).float()

    data2_triples = test_case2_data.batch  # [:use_data2]]
    data2_targets = test_case2_data.multi_class_labels  # [:use_data2]]

    data2_tf = test_case2_data.get_real_labels(data2_triples)  # [:use_data2]]

    data2_sentences = [construct_questions(kg, triple, llm_question_cache, llm_answer_cache, ty=model_type)[0] for
                       triple in
                       tqdm(data2_triples)]  # [:use_data2]]

    data2_logits, _ = evaluate_on_dataset(data2_sentences, torch.tensor(data2_targets),
                                          evaluation_model,
                                          device, 0, batch_size=batch_size,
                                          max_seq_len=max_seq_len, train=False, n_classes=n_classes,
                                          true_false=data2_tf, autocast_dtype=torch.float16, get_logits=True)
    data1_logits = torch.softmax(data1_logits, dim=1)
    data2_logits = torch.softmax(data2_logits, dim=1)
    data2_targets = torch.tensor(data2_targets)
    # repeat 100 times for data2_logits and data2_targets
    data2_logits = data2_logits.repeat(10000, 1)
    data2_targets = data2_targets.repeat(10000)

    ranges = []
    for target_class in range(3):
        ranges.append(logit_estimation(data1_llm_logits[:, target_class], data1_logits[:, target_class],
                                       len(data2_logits[:, target_class])))

    print('confidence levels', ranges)
    min_accuracy_logits = data2_logits.clone().cpu()

    for target_class in range(3):
        for i in range(len(data2_logits)):
            if data2_targets[i] == target_class:
                min_accuracy_logits[i, target_class] += ranges[target_class][0]
            else:
                min_accuracy_logits[i, target_class] += ranges[target_class][1]

    # min accuracy
    max_accuracy_logits = data2_logits.clone().cpu()
    for target_class in range(3):
        for i in range(len(data2_logits)):
            if data2_targets[i] == target_class:
                max_accuracy_logits[i, target_class] += ranges[target_class][1]
            else:
                max_accuracy_logits[i, target_class] += ranges[target_class][0]

    max_accuracy_preds = torch.argmax(max_accuracy_logits, dim=1)
    min_accuracy_preds = torch.argmax(min_accuracy_logits, dim=1)
    from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
    print('original:')
    preds = torch.argmax(data2_logits, dim=1)
    print('f1', f1_score(data2_targets, preds, average='macro'))
    print('precision', precision_score(data2_targets, preds, average='macro'))
    print('recall', recall_score(data2_targets, preds, average='macro'))
    print('confusion matrix', confusion_matrix(data2_targets, preds))
    # preds vs data2_targets
    print('min_accuracy:')
    print('f1', f1_score(data2_targets, min_accuracy_preds, average='macro'))
    print('precision', precision_score(data2_targets, min_accuracy_preds, average='macro'))
    print('recall', recall_score(data2_targets, min_accuracy_preds, average='macro'))
    print('confusion matrix', confusion_matrix(data2_targets, min_accuracy_preds))

    print('max_accuracy:')
    print('f1', f1_score(data2_targets, max_accuracy_preds, average='macro'))
    print('precision', precision_score(data2_targets, max_accuracy_preds, average='macro'))
    print('recall', recall_score(data2_targets, max_accuracy_preds, average='macro'))
    print('confusion matrix', confusion_matrix(data2_targets, max_accuracy_preds))


if __name__ == '__main__':
    pass
