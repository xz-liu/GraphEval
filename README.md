# GraphEval: Evaluating the Factuality of Large Language Models using Large-Scale Knowledge Graphs
![](https://img.shields.io/badge/version-1.0.0-blue)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/xz-liu/GraphEval/blob/master/LICENSE)
[![arxiv badge](https://img.shields.io/badge/arxiv-2404.00942-red)](https://arxiv.org/abs/2404.00942)
[![Pytorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white)](https://pytorch.org/)


> We propose **GraphEval** to evaluate an LLM's performance using a substantially large test dataset. Specifically, the test dataset is retrieved from a large knowledge graph with more than 10 million facts without expensive human efforts. Unlike conventional methods that evaluate LLMs based on generated responses, **GraphEval** streamlines the evaluation process by creating a judge model to estimate the correctness of the answers given by the LLM. Our experiments demonstrate that the judge model's factuality assessment aligns closely with the correctness of the LLM's generated outputs, while also substantially reducing evaluation costs.  Besides, our findings offer valuable insights into LLM performance across different metrics and highlight the potential for future improvements in ensuring the factual integrity of LLM outputs.

<div align="center">
    <img src="https://github.com/xz-liu/GraphEval/blob/master/assets/framework.jpg" width="95%" height="auto" />
</div>


## 🔬 Dependencies
```bash
pip install -r requirements.txt
```
#### Details
- Python (>= 3.7)
- [PyTorch](http://pytorch.org/) (>= 1.13.1)
- numpy (>= 1.19.2)
- [Transformers](http://huggingface.co/transformers/) (== 4.38.2)


## 📚 Data Preparation

Please download `mappingbased-objects_lang=en.ttl.bzip2` from the DBpedia dataset and unzip it. A program argument is provided to specify the path to the file.

DBpedia dataset can be downloaded from [here](https://www.dbpedia.org/). 


## 🚀 Running the code

The 3 steps in the papers are implemented in the following files:

1. `collect.py`
2. `train.py`
3. `eval.py`

The code provides arguments to specify settings, paths, and hyperparameters. To see the arguments, run the following command:

```bash
python collect.py --help
```
Here, you can use any of the `collect.py`, `train.py`, and `eval.py` files to run the help command.

## 🤝 Cite:
Please consider citing this paper if you use the ```code``` or ```data``` from our work.
Thanks a lot :)

```bigquery
@journal{liu2024evaluating,
      title={Evaluating the Factuality of Large Language Models using Large-Scale Knowledge Graphs}, 
      author={Xiaoze Liu and Feijie Wu and Tianyang Xu and Zhuo Chen and Yichi Zhang and Xiaoqian Wang and Jing Gao},
      year={2024},
      journal={arXiv preprint arXiv:2404.00942}
}
```

