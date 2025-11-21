# GPT-RE-FT

Code for reproducing the GPT-RE with fine-tuned RE representations, EMNLP 2023 paper "[GPT-RE: In-context Learning for Relation Extraction using Large Language Models](http://simaclanthology.org/2023.emnlp-main.214/)". For better code readability, this repo only includes GPT-RE-FT, referring to [GPT-RE](https://github.com/YukinoWan/GPT-RE) for other methods.
```
GPT-RE-FT
 |-- GPT-RE (Retrieval-Based In-Context Learning)
 |-- REbaseline (Training Representation Models)
```

## Requirements
* torch >= 1.8.1
* transformers >= 3.4.0
* wandb
* ujson
* tqdm
* faiss
* openai

## Dataset
Please refer [GPT-RE](https://github.com/YukinoWan/GPT-RE) or [PURE](https://github.com/princeton-nlp/PURE) to prepare the datasets. The expected structure of files is:
```
RE_improved_baseline
 |-- data
 |    |-- semeval
 |    |    |-- train.json        
 |    |    |-- dev.json
 |    |    |-- test.json
 |    |-- scierc
 |    |    |-- train.json        
 |    |    |-- dev.json
 |    |    |-- test.json
```

## Training The RE Representation Model (REbaseline)
To train the representation models, we adopt an efficent implementation of entity-marker based method "[An Improved Baseline for Sentence-level Relation Extractions](https://aclanthology.org/2022.aacl-short.21/)". The commands and hyper-parameters for running experiments can be found in the ``scripts`` folder.
```bash
>> sh run_bert_ace.sh
>> sh run_bert_scierc.sh
>> sh run_bert_semeval.sh
```
###  Checkpoints

| Dataset  | Model       | Download |
| -------- | ------------------ | -------- |
| Semeval | bert-base-uncased             |  [link](https://drive.google.com/file/d/18oQjWl8_i3v7QPvk6GnE1eUsVEzNmA3j/view?usp=sharing)
| SciERC     | scibert_scivocab_uncased             |  [link](https://drive.google.com/file/d/12p5E2cD0rFavwbvUUmBDZQDxNiyzxYyt/view?usp=sharing)
| ACE     | bert-base-uncased             |  [link](https://drive.google.com/file/d/1RhDabvsvc7INZQZgU-4OTxd6KKcAqtJU/view?usp=sharing)

## Retrieval-Based In-Context Learning (GPT-RE)
```bash
>> sh run_relation.sh
```
An example for knn-retrieved demonstrations could be found in results/knn-semeval.

## Acknowledgements
Our code is based on [GPT-RE](https://github.com/YukinoWan/GPT-RE) and [RE-improved-baseline](https://github.com/wzhouad/RE_improved_baseline/tree/main)