# mrc-level2-nlp-16

## Table of Contents
  1. [Project Overview](#Project-Overview)
  2. [Getting Started](#Getting-Started)
  3. [Hardware](#Hardware)
  3. [Code Structure](#Code-Structure)
  4. [Detail](#Detail)

## Project Overview
  * 목표

  * 모델

  * Data

  * Result

  * Contributors
    * 김아경([github](https://github.com/EP000)): 
    * 김현욱([github](https://github.com/powerwook)): 
    * 김황대([github](https://github.com/kimhwangdae)): 
    * 박상류([github](https://github.com/psrpsj)): 
    * 정재현([github](https://github.com/JHyunJung)): 
    * 최윤성([github](https://github.com/choi-yunsung)): 

## Getting Started
  * Install requirements
    ``` bash
      # AEDA를 사용하기 위한 jdk 설치
      apt install default-jdk
      
      # requirement 설치
      pip install -r requirements.txt 
    ```
  * Train model
    ``` bash

    ```
  * Inference Model

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
    └── test_dataset/
```

## Detail



