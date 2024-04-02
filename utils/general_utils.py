import time
from typing import Tuple, List, Optional, Dict, Any
from tqdm import tqdm,trange
import os
from functools import wraps, partial
import torch
from contextlib import nullcontext
from args import args
class SharedStringList:
    def __init__(self, strings):
        encoded_strings = [s.encode('utf-8') for s in strings]
        self.buffer_size = sum(len(s) for s in encoded_strings) + len(strings)  
        self.tensor = torch.zeros(self.buffer_size, dtype=torch.uint8) 

        self.indexes = [0]

        offset = 0
        for s in tqdm(encoded_strings, desc="Encoding strings into shared memory"):
            self.tensor[offset:offset + len(s)] = torch.tensor(list(s), dtype=torch.uint8)
            offset += len(s)
            self.tensor[offset] = 0  
            offset += 1
            self.indexes.append(offset)

    def __getitem__(self, index):
        start = self.indexes[index]
        end = self.indexes[index + 1] - 1  
        return bytes(self.tensor[start:end].tolist()).decode('utf-8')

    def share_memory(self):
        self.tensor.share_memory_()

    def __len__(self):
        return len(self.indexes) - 1
def exact_match_score(gold: str, pred: str) -> int:
    return int(gold == pred)


def get_choice(pred) -> str:
    words = pred.split()
    for word in words:
        # remove characters other than alphabet, numbers
        word = ''.join(e for e in word if e.isalnum())
        if len(word) == 0:
            continue
        if word[0].isupper() and len(word) == 1 and word in 'ABCDE':
            return word[0]
    return "W"


def option_match(pred: str, gold: str) -> int:
    if len(pred) == 0:
        return 0
    print('The choice is ' + get_choice(pred) + ' and the gold is ' + gold)
    return int(get_choice(pred) == gold)


class BERTSimScore:
    def __init__(self, threshold=0.8, device='cuda'):
        from sentence_transformers import SentenceTransformer, util
        self.model = SentenceTransformer('bert-base-nli-mean-tokens').to(device)
        self.threshold = threshold

    def __call__(self, gold: str, pred: str) -> int:
        from sentence_transformers import SentenceTransformer, util
        embeddings = self.model.encode([gold, pred])
        cosine_scores = util.pytorch_cos_sim(embeddings[0], embeddings[1])
        return int(cosine_scores > self.threshold)


def timeout_wrapper(func, args, kwargs, timeout_duration=10, default=None):
    import signal

    class TimeoutError(Exception):
        pass

    def handler(signum, frame):
        raise TimeoutError()

    # set the timeout handler
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout_duration)
    try:
        result = func(*args, **kwargs)
    except TimeoutError as exc:
        result = default
    finally:
        signal.alarm(0)
    if result is None:
        # throw error
        raise TimeoutError()

    return result


def plot_dummy_separator():
    import matplotlib.pyplot as plt
    import numpy as np

    # show a random chart with matplotlib
    x = np.random.rand(100)
    y = np.random.rand(100)
    plt.scatter(x, y)
    plt.title("Random scatterplot")
    plt.show()


def partition_graph(kg, save_path, node_per_partition=10000):
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(len(kg.entitynames)))
    G.node = G.nodes
    G.edge = G.edges
    h2t = [
        (int(h), int(t), 1)
        for h, r, t in kg.triples]

    # add edges
    print("Adding edges to graph...")
    G.add_weighted_edges_from(h2t)
    # perform metis partitioning
    print("Performing metis partitioning...")
    import nxmetis

    partition_number = len(kg.entitynames) // node_per_partition
    print("Partition number: ", partition_number)
    (edgecuts, parts) = nxmetis.partition(G, partition_number)

    print("Edgecuts: ", edgecuts)

    partitions = []
    for pt in tqdm(parts):
        partitions.append([kg.id2entity[pt[i]] for i in range(len(pt))])

    print("Save partitions to file...")
    import torch
    torch.save(partitions, save_path)


def get_entity2part(kg, partitions):
    entity2part = {}
    for i, pt in enumerate(partitions):
        for entity in pt:
            entity2part[kg.entity2id[entity]] = i
    return entity2part


def get_triple2part(kg, triples, entity2part):
    triple2part = {}
    for h, r, t in triples:
        h, r, t = int(h), int(r), int(t)
        if entity2part[h] == entity2part[t]:
            triple2part[(h, r, t)] = entity2part[h]
    return triple2part


def get_part2triple(triples2part):
    part2triples = {}
    for triple, part in triples2part.items():
        if part not in part2triples:
            part2triples[part] = []
        part2triples[part].append(triple)
    return part2triples


def save_to(path, content, backend='torch'):
    """
    Saves the given content to a file at the specified path.
    Creates the directory if it does not exist.

    :param path: The path where the file will be saved.
    :param content: The content to be written to the file.
    """

    time_now= time.perf_counter()
    if args.no_save:
        print('***** Not saving to file', path, 'because no_save is set to True')
        return
    # Extract the directory from the path
    directory = os.path.dirname(path)

    # Check if the directory exists, if not create it
    if directory != '' and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory at {directory}")

    if backend == 'torch':
        import torch
        torch.save(content, path)
    elif backend == 'pickle':
        import pickle
        with open(path, 'wb') as file:
            pickle.dump(content, file)
    else:
        # Write the content to the file
        with open(path, 'w') as file:
            file.write(content)

    print(f"File saved successfully at {path}, time taken: {time.perf_counter()-time_now} seconds")


def file_exists(path):
    """
    Checks if a file exists at the specified path.

    :param path: The path where the file is located.
    :return: True if the file exists, False otherwise.
    """
    return os.path.exists(path)


def load_from(path, backend='torch', verbose=True):
    """
    Loads the content from the file at the specified path.

    :param path: The path where the file is located.
    :return: The content of the file.
    """
    time_now= time.perf_counter()
    # first, check if the file exists
    if path is None:
        raise FileNotFoundError(f"Path is None")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found at {path}")
    if verbose:
        print(f"File found at {path}, loading...")


    if backend == 'torch':
        import torch
        f= torch.load(path)
    elif backend == 'pickle':
        import pickle
        with open(path, 'rb') as file:
            f= pickle.load(file)
    elif backend == 'json':
        import json
        with open(path, 'r') as file:
            f= json.load(file)
    else:
        # Read the content from the file
        with open(path, 'r') as file:
            f= file.read()

    if verbose:
        print(f"File loaded successfully from {path}, time taken: {time.perf_counter()-time_now} seconds")
    return f
