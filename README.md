# IntraMix
This is the official implementation of our NeurIPS 2024 Paper: IntraMix: Intra-Class Mixup Generation for Accurate Labels and Neighbors. [paper](https://arxiv.org/abs/2405.00957)

# Requirements

```
python ==3.8
torch ==1.9.1
torch-cluster ==1.5.9
torch-geometric==2.3.0
torch-scatter ==2.1.0
torch-sparse ==0.6.15
numba  ==0.58.1                   
numpy  ==1.24.3                    
ogb  == 1.3.6
optuna ==3.5.0
```

# Datasets

In the experimental process, we used a total of seven datasets: Cora, CiteSeer, Pubmed, CS, PHYSICS, ogbn-arxiv, and Flickr. These datasets are publicly available, and you can find them on the official websites or download them directly through PyG. After downloading, place the dataset files in ./data. 

For convenience, the following are the download links for each dataset. 

Cora，CiteSeer，Pubmed：https://github.com/kimiyoung/planetoid/raw/master/data

CS,PHYSICS: https://github.com/shchur/gnn-benchmark/raw/master/data/npz/

ogbn-arxiv：https://snap.stanford.edu/ogb/data/nodeproppred/

Flickr：https://snap.stanford.edu/data/web-flickr.html

# Train

The main running file is main.py, and we have organized a set of default parameters for GNNs and corresponding datasets. When running with default parameters, use the following code:

```
python main.py --dataset_name [dataset_name] --model_name [model_name] --cuda_device [cuda_device]
```

## Optim

If you find the given default parameters perform poorly, you can run the optim.py file to search for optimal parameters using the AutoML tool. Use the following code:

```
python optim.py --dataset_name [dataset_name] --model_name [model_name] --cuda_device [cuda_device] --n_trials [n_trials]
```

To ensure generalizability across different datasets and GNNs, the default hyperparameter search space is extensive, covering almost all possible scenarios. For instance, dropout rate ranges from 0 to 0.99. This might result in a lengthy search time and suboptimal results. If you have more suitable parameter ranges for a specific dataset and model, you can modify the corresponding ranges in lines 66~77 of the optim.py file.

# Reference
```
@article{zheng2024intramix,
  title={IntraMix: Intra-Class Mixup Generation for Accurate Labels and Neighbors},
  author={Zheng, Shenghe and Wang, Hongzhi and Liu, Xianglong},
  journal={arXiv preprint arXiv:2405.00957},
  year={2024}
}
```