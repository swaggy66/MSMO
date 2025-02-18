# MSMO

This repo contains the data and code for our paper ****Multi-Scale and Multi-Objective Optimization for Cross-Lingual Aspect-Based Sentiment Analysis****.

## Requirements

- torch==1.3.1
- numpy==1.19.4
- transformers==3.4.0 
- sentencepiece==0.1.91
- tokenizer==0.9.2
- sacremoses==0.0.43

## Quick Start 

- Download the data and place it in the data/ folder.
- Download the pre-trained multilingual language model mBERT or XLM-R
- To quickly reproduce the results you can run the following setting:

```
bash run.sh 
```

## Usage

To run experiments under different settings, change the exp_type setting:

- supervised refers to the supervised setting
- macs_kd: multilingual distillation

# Citation

If the code is used in your research, please star our repo and cite our paper as follows:

```

```
