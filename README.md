# mrc-level2-nlp-16

## Team Wrap-up Report
  * [notion](https://glossy-crib-1b1.notion.site/_level2_mrc_16-fef387ea8c734952a9eacf4ddb78c707)

## Table of Contents
  1. [Project Overview](#Project-Overview)
  2. [Getting Started](#Getting-Started)
  3. [Hardware](#Hardware)
  3. [Code Structure](#Code-Structure)
  4. [Detail](#Detail)

## Project Overview
  * 목표
    * 사전에 구축되어있는 Knowledge resourse에서 질문의 답을 찾는 Open-Domain Question Answering(ODQA) 구축
  * 모델
    * Retriever: COIL + ElasticSearch (query+context 150개)
    * Reader: SoftVoting(conv + lstm_conv + rnn_conv + base + bert_base(conv))

  * Data
    * train_dataset: 3952개 train set/ 240개 Validation set
    * test_dataset: 240개 public validatoin set / 360개 private validation set  
    * Wikipedia_dataset

  * Result
    * Public
      * Exact Match: 70.000
      * F1 Score: 79.490
    * Private
      * Exact Match: 64.170
      * F1 Score: 75.810

  * Contributors
    * 김아경([github](https://github.com/EP000)): EDA, Negative Sampling, Post-Porcessing
    * 김현욱([github](https://github.com/powerwook)): Elastic-search
    * 김황대([github](https://github.com/kimhwangdae)): Retriever (DPR, COIL, Retriever , Retriver ensemble, Elasticsearch ensemble), Data Augmentation && Reader(Train Dataset Negative sampling 후, Reader 학습 진행), contexts joining delimiter 실험 
    * 박상류([github](https://github.com/psrpsj)): K-Fold 구현, Ensemble 구현, Pre-processing, Post-Processing 실험
    * 정재현([github](https://github.com/JHyunJung)): ElasticSearch , Addquery 제작
    * 최윤성([github](https://github.com/choi-yunsung)): Reader (custom_layer, qestion_token span masking), Retriever (BM25, COIL), Data Augmentation (question generation, backtranslation), Ensemble (soft-votting) 구현 및 실험

## Getting Started
  * Install requirements
    ``` bash
      # AEDA를 사용하기 위한 jdk 설치
      apt install default-jdk
      
      # requirement 설치
      bash ./install/install_requirements.sh
    ```
  * Train model
    ``` bash
    python train.py --output_dir ./models/train_dataset --do_train [if use K-fold add --do_kfold]
    ```
  * Inference Model
    ``` bash
    python inference.py --output_dir ./outputs/test_dataset/ --dataset_name ../data/test_dataset/ --model_name_or_path ./models/train_dataset/ --do_predict [if use K-fold add --do_kfold]
    ```
## Hardware
The following specs were used to create the original solution.
- Ubuntu 18.04.5 LTS
- Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz
- NVIDIA Tesla V100-SXM2-32GB

## Code Structure
```
├── code/      
│   ├── install/
│   │   └── install_requirements.sh
│   │
│   ├── reader/
│   │   ├── ConvModel.py
│   │   ├── LSTMConvModel.py
│   │   ├── LSTMModel.py
│   │   ├── RNNConvModel.py
│   │   └── models.py
│   │
│   ├── retriever/
│   │   ├── coil/
│   │   │   ├── data_helper/
│   │   │   │   ├── build_train_from_triplet.py
│   │   │   │   └── make_triplet.py
│   │   │   ├── retrieve/
│   │   │   │   ├── retriever_ext/
│   │   │   │   │   ├── scatter.pyx
│   │   │   │   │   └── setup.py
│   │   │   │   ├── format-query.py
│   │   │   │   ├── merger.py
│   │   │   │   ├── retriever-fast.py
│   │   │   │   └── sharding.py
│   │   │   ├── arguments.py
│   │   │   ├── macro_datasets.py
│   │   │   ├── modeling.py
│   │   │   ├── run_macro.py
│   │   │   ├── score_to_macro.py
│   │   │   ├── trainer.py
│   │   │   └── coil_tutorial.md
│   │   │
│   │   ├── elastic_search/
│   │   │   └── elasticsearch_retriever.md
│   │   │
│   │   ├── bm25.py
│   │   ├── coil.py
│   │   └── tfidf.py
│   │
│   ├── arguments.py
│   ├── argumentation.py
│   ├── inference.py
│   ├── postprocess.py
│   ├── preprocess.py
│   ├── train.py
│   └── trainer_qa.py                   
│
└── data/
    ├── train_dataset/
    ├── test_dataset/
    └── wikipedia_documents.json
```





