# GPT-RE-FT

Code for reproducing the GPT-RE with fine-tuned RE representations, EMNLP 2023 paper "[GPT-RE: In-context Learning for Relation Extraction using Large Language Models](http://simaclanthology.org/2023.emnlp-main.214/)". For better code readability, this repository includes only the GPT-RE-FT implementation; please refer to [GPT-RE](https://github.com/YukinoWan/GPT-RE) for the remaining methods.
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
Please refer to [GPT-RE](https://github.com/YukinoWan/GPT-RE) or [PURE](https://github.com/princeton-nlp/PURE) for preparing the datasets. The expected structure of files is:
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
To train the representation models, we adopt an efficient implementation of entity-marker based method "[An Improved Baseline for Sentence-level Relation Extractions](https://aclanthology.org/2022.aacl-short.21/)". The commands and hyper-parameters for running experiments can be found in the ``scripts`` folder.
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
Please refer to [GPT-RE](https://github.com/YukinoWan/GPT-RE) for detailed description of augments.
An example for knn-retrieved demonstrations could be found in results/knn-semeval.

## Citation

If you find this work helpful, please cite our paper:

```bibtex
@inproceedings{wan2023gpt,
  title={GPT-RE: In-context Learning for Relation Extraction using Large Language Models},
  author={Wan, Zhen and Cheng, Fei and Mao, Zhuoyuan and Liu, Qianying and Song, Haiyue and Li, Jiwei and Kurohashi, Sadao},
  booktitle={Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  year={2023}
}
@inproceedings{han-etal-2025-amr,
    title = "{AMR}-{RE}: {A}bstract {M}eaning {R}epresentations for Retrieval-Based In-Context Learning in Relation Extraction",
    author = "Han, Peitao  and
      Pereira, Lis  and
      Cheng, Fei  and
      She, Wan Jou  and
      Aramaki, Eiji",
    booktitle = "Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 4: Student Research Workshop)",
    year = "2025",
}
```

## Acknowledgements
Our code is based on [GPT-RE](https://github.com/YukinoWan/GPT-RE) and [RE-improved-baseline](https://github.com/wzhouad/RE_improved_baseline/tree/main)
