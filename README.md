# MSMO

This repo contains the data and code for our paper ****Multi-Scale and Multi-Objective Optimization for Cross-Lingual Aspect-Based Sentiment Analysis****.

[![arXiv](https://img.shields.io/badge/arXiv-2502.13718-b31b1b.svg)](https://arxiv.org/abs/2502.13718)


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
bash run_absa.sh 
```

## Usage

To run experiments under different settings, change the exp_type setting:

- supervised refers to the supervised setting
- macs_kd: multilingual distillation

# Citation

If the code is used in your research, please star our repo and cite our paper as follows:

```
@misc{wu2025multi,
      title={Multi-Scale and Multi-Objective Optimization for Cross-Lingual Aspect-Based Sentiment Analysis}, 
      author={Chengyan Wu and Bolei Ma and Ningyuan Deng and Yanqing He and Yun Xue},
      year={2025},
      eprint={2502.13718},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.13718}, 
}
```
