#!/bin/bash
### install requirements for pstage3 baseline
# pip requirements
pip install datasets==1.5.0
pip install transformers==4.11.3
pip install tqdm==4.41.1
pip install pandas==1.1.4
pip install scikit-learn==0.24.1
pip install konlpy==0.5.2

# faiss install (if you want to)
pip install faiss-gpu

pip install rank_bm25
pip install pororo
pip install elasticsearch

pip install koeda
apt install default-jdk
pip install elasticsearch

# install for coil
pip install Cython
apt-get install g++
pip install fuzzywuzzy
