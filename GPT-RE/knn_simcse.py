from shared.prompt import instance
from transformers import pipeline
import numpy as np

def find_lmknn_example(gpu_index_flat, test_dict, train_dict, train_sentences, k):
    
    test_sentence = instance(test_dict).lm_mask
    extractor = pipeline(model="roberta-large", task="feature-extraction")
    result = extractor(test_sentence, return_tensors=True)
    
    embed = result.detach().numpy().copy()
    xq = np.array([embed[0][-3]])

    print(xq.shape)
    D, I = gpu_index_flat.search(xq, k)
    print(I)

    knn_list = [train_dict[train_sentences[i]] for i in I[0,:k]]

    return knn_list

def find_ftknn_example(gpu_index_flat, test_dict, train_dict, k):
    
    embed = test_dict['feature']
    D, I = gpu_index_flat.search(embed.reshape(1, -1), k)
    print(I)

    knn_list = [train_dict[i] for i in I[0,:k]]
    # print([i['relations'] for i in knn_list])
    # print(test_dict)
    return knn_list