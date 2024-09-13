# Self-Supervised Learning for Tabular Data

<a href="https://pytorch.org/get-started/locally/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.7--3.9-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 2.3.1+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning 2.3.3+-792ee5?style=for-the-badge&logo=pytorchlightning&logoColor=white"></a>
<a href="https://ml.azure.com/"><img alt="MLAzure" src="https://img.shields.io/badge/azure-%230072C6.svg?style=for-the-badge&logo=microsoftazure&logoColor=white"></a>

## 📌&nbsp;&nbsp;Introduction
This repository reproduces the MET (Masked Encoding for Tabular Data) framework for self-supervised representation learning with tabular data.

Original paper: https://table-representation-learning.github.io/assets/papers/met_masked_encoding_for_tabula.pdf

![MET](https://github.com/user-attachments/assets/503624cd-71af-41b5-941c-249f19ef1eed)



Original repo: https://github.com/google-research/met

### Why Self-Supervised Learning? (Upstream)
In the real world, data are noisy. According to IBM, the data usually contain more than 40% of noises. The model, as a result, tends to learn the incorrect labels. Therefore, instead of training the model with labels, self-supervised learning trains the model to learn representations of the input coordinates without labels. This is the reason why we decided to use the self-supervised learning as our upstream step.

### why MET? (Upstream)
MET masks randomly N% of the input coordinates in every batch. This gives more randomness while training, and makes the model performance stabiler, especially when the input data contain high level of noises.

### Why PyTorch?
The Original Paper(MET) was written in tensorflow. But the most widely used library is PyTorch when it comes to deep learning. We also used PyTorch Lightning and Hydra to increase the modulity and reusability of our code.

### Why PyTorch Lightning?
[PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) is a lightweight PyTorch wrapper for high-performance AI research.
It makes your code neatly organized and provides lots of useful features, like ability to run model on CPU, GPU, multi-GPU cluster and TPU.

### Why Hydra?
Hydra enables us to override the configurations without modifying the code. It reduces the unnecessary complexity of work while fine-tuning. 

### Finetuning the target ML model (Downstream)
Using the embeddings extracted by the Upstream model, we fine-tuned another ML model to get the final predicted classification labels. For simplicity and efficient comparision, we picked the the Logistic regression as our base model and tracked the effect of Pre-Training.


## 📌&nbsp;&nbsp;Code Structure

```
.
├── README.md
├── __init__.py
├── requirements.txt
├── configs/
    ├── configs.yaml
    ├── experiments/
        └── msk_50.yaml
├── data/
├── notebooks/
    ├── data_prep/
    ├── met_eval/
    ├── traditional_ml/
├── outputs/
    ├── pretrainlocal/
├── wandb/
└── src/
    ├── __init__.py
    ├── train_classifier.py
    ├── train_module.py
    ├── datamodules/
        ├── __init__.py
        ├── tabular_datamodule.py
        ├── datasets/
            ├── __init__.py
            ├── tabular_dataset.py
    ├── models/
        ├── __init__.py
        ├── lr_classifier.py
        ├── tranformer_autoencoder.py
        ├── modules/
            ├── old.py
            ├── autoencoders/
                ├── transformer/
                    ├── decoder.py
                    ├── encoder.py
                    ├── masking.py
                    ├── positional_embedding.py
                    ├──transformer_block.py
    └── utils/
            ├── utils_funcs.py
```

## 📌&nbsp;&nbsp;How to run our code?
You can do pre-training experiments by configuring the hyper-parameters. Under folder configs/pretrain/, you can find a general configs.yaml, which contains the default setting of masking rates and also other hyper-parameters. If you want want to modify something, you can change them in .yaml files under the experments folder, as you can find in the example files under /experiments folder. Thanks to the python framework Hydra, you don't need to re-write every settings in the .yaml file. You can simply write the parts you want to fine-tune.

Then, you train the upstream.

To run the upstream model:
```
python /src/train_module.py +config_file=path/to/your/config  
```

Afterwards, you train the downstream using the extracted embeddings. 

To run the downstream model:
```
python /src/train_classifier.py +config_file=path/to/your/config
```

## 📌&nbsp;&nbsp;Blog Link

I wrote this  [Tutorial][https://medium.com/@khyemin/self-supervised-learning-for-tabular-data-3148cd32fcf8] as a guide into MET and our repository. 
2nd Blog Post is foccusing on the practical implementation of self supervised learning algorithm [https://medium.com/@chaturvedivikas059/mastering-noisy-data-a-deep-dive-into-self-supervised-learning-with-transformers-for-tabular-data-fad582590d4c]

## 📌&nbsp;&nbsp;Contributors
Hyemin Kim: hyemin.kim@telekom.de
Felipe Villa-Arenas: luis-felipe.villa-arenas@telekom.de
Vikas Kumar: vikas.kumar@t-systems.com
