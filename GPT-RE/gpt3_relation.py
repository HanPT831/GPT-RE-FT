import json
import os
import pandas as pd
import argparse
import faiss
import sys
import math
import time
import torch
from tqdm import tqdm
from gpt3_api import Demo
import random
import numpy as np
from itertools import combinations
from testeval import compute_f1
from shared.const import semeval_reltoid
from shared.const import semeval_idtoprompt
from shared.const import ace05_reltoid
from shared.const import ace05_idtoprompt
from shared.const import tacred_reltoid
from shared.const import scierc_reltoid
from shared.const import wiki_reltoid
from shared.prompt import instance
from sklearn.metrics import classification_report
from knn_simcse import find_lmknn_example, find_ftknn_example

from shared.prompt import generate_select_prompt
from shared.prompt import generate_select_auto_prompt
from shared.result import get_results_onebyone
from shared.result import get_results_select

from re_models.model import REModelforKnn
from re_models.prepro import SemevalProcessor, AceProcessor, AceProcessor
from re_models.utils import set_seed, collate_fn

from transformers import AutoConfig, AutoTokenizer
from torch.utils.data import DataLoader

def generate_relation_dict_label(dataset):
    labels = []
    with open(dataset, "r") as f:
        relation_dict = {}
        for line in f.read().splitlines():
            tmp_dict = json.loads(line)

            
            if tmp_dict["relations"]== [[]]:
                rel = "None"
            else:
                rel = tmp_dict["relations"][0][0][4]
            if rel not in relation_dict.keys():
                relation_dict[rel] = len(relation_dict.keys())
            
            labels.append(relation_dict[rel])

        print(relation_dict)
        return relation_dict, labels
    
def generate_label(dataset, relation_dict):
    labels = []
    with open(dataset, "r") as f:
        for line in f.read().splitlines():
            tmp_dict = json.loads(line)

            
            if tmp_dict["relations"]== [[]]:
                rel = "NONE"
            else:
                rel = tmp_dict["relations"][0][0][4]
            if rel not in relation_dict.keys():
                relation_dict[rel] = len(relation_dict.keys())
            
            labels.append(relation_dict[rel])

        print(relation_dict)
        return labels

def generate_query(h_type, t_type, relation_list, query_dict):
    query_list = []
    #print(query_dict)
    for rel in relation_list:
        if rel == "None":
            continue
        else:
            query = query_dict[str((h_type,rel,t_type))]
            query_list.append(query)
    return query_list

def build_query_dict(dataset):
    with open("query_templates/ace2005.json", "r") as f:
        whole_dict = json.load(f)
        query_dict = whole_dict["qa_turn2"]
        return query_dict
    
def get_example(example_path,reltoid):
    example_dict = {k:list() for k in reltoid.values()}
    with open(example_path, "r") as f:
        for index, line in enumerate(f.read().splitlines()):
            tmp_dict = json.loads(line)
            if tmp_dict["relations"] == [[]]:
                rel = "NONE"
                example_dict[reltoid[rel]].append(tmp_dict)
            else:
                rel = tmp_dict["relations"][0][0][4]
                example_dict[reltoid[rel]].append(tmp_dict)
    return example_dict

def get_sample(example_path):
    examples = []
    with open(example_path, "r") as f:
        for index, line in enumerate(f.read().splitlines()):
            tmp_dict = json.loads(line)
            examples.append(tmp_dict)
    return examples
     

def auto_generate_example(example_dict, reltoid, idtoprompt, num_per_rel, num_na, random_label, reasoning, demo):
    num_example = num_per_rel * (len(example_dict.keys()) - 1)

    examples = []
    for relid in example_dict.keys():
            examples.append(random.sample(example_dict[relid], num_per_rel))
            

    flat_examples = [item for sublist in examples for item in sublist]
    #print(len(examples))
    example_list = random.sample(flat_examples, num_example)
    #assert False
    
    example_prompt = str()
    for tmp_dict in example_list:
        string = " ".join(tmp_dict["sentences"][0])
        sub_head = tmp_dict["ner"][0][0][0]
        sub_tail = tmp_dict["ner"][0][0][1] + 1

                
        obj_head = tmp_dict["ner"][0][1][0]
        obj_tail = tmp_dict["ner"][0][1][1] + 1

        entity1 = " ".join(tmp_dict["sentences"][0][sub_head:sub_tail])
        entity1_type = tmp_dict["ner"][0][0][2]
        entity2 = " ".join(tmp_dict["sentences"][0][obj_head:obj_tail])
        entity2_type = tmp_dict["ner"][0][1][2]

        if random_label:
            rel = random.choice([x for x in reltoid.keys()])
        elif tmp_dict["relations"] == [[]]:
            rel = 'NONE'
        else:
            rel = tmp_dict["relations"][0][0][4]


        if not reasoning:
            prompt_query = "\nContext: " + string + "\n" + "Given the context, the relation between " + entity1 + " and " + entity2 + " is " + idtoprompt[reltoid[rel]] + ".\n"
        else:
            #tmp_query = "\nGiven the sentence: \"" + string + "\", What are the clues that lead the relation between \"" + entity1 + "\" and \"" + entity2 + "\" to be " + idtoprompt[reltoid[rel]] + "?"
            tmp_query = "What are the clues that lead the relation between \"" + entity1 + "\" and \"" + entity2 + "\" to be " + idtoprompt[reltoid[rel]] + " in the sentence \"" + string + "\"?"
            prompt_query = "\nContext: " + string + "\n" + "Given the context, the relation between " + entity1 + " and " + entity2 + " is " + idtoprompt[reltoid[rel]] + ". It is because:"

            results, probs = demo.get_multiple_sample(tmp_query)
            prompt_query = prompt_query + results[0] +"\n"
        example_prompt += prompt_query
    return example_prompt



def find_prob(target, result, probs):
    if False:
        print(result)
        print("targettarget\n")
        print(target)
        print("tokentoken\n")
        print(probs["tokens"])
        print("===============\n")
    try:
        index = [x.strip() for x in probs["tokens"]].index(str(target))
        return math.exp(probs["token_logprobs"][index])
    except:
        len_target = len(target)
        for i in range(2, len_target+1):
            for j in range(len(probs["tokens"])):
                if i + j > len(probs["tokens"]):
                    continue
                tmp_word = "".join([probs["tokens"][x] for x in range(j, j+i)])
                if tmp_word.strip() != target:
                    continue
                else:
                    start = j
                    end = j + i
                    sum_prob = 0
                    for k in range(start, end):
                        sum_prob += math.exp(probs["token_logprobs"][k])
                    return sum_prob / i
        return 0.0

def smooth(x):
    if True:
        return np.exp(x)/sum(np.exp(x)) 
    else:
        return x

def generate_lm_example(gpu_index_flat, tmp_dict, train_list, model, k, reltoid, idtoprompt, reasoning, demo,args):
    example_list = find_ftknn_example(gpu_index_flat, tmp_dict, train_list, k)
    
    if args.reverse:
        example_list.reverse()
    label_other = 0
    tmp_knn = []
    example_prompt = str()

    for tmp_dict in example_list:
        string = " ".join(tmp_dict["sentences"][0])
        sub_head = tmp_dict["ner"][0][0][0]
        sub_tail = tmp_dict["ner"][0][0][1] + 1

                
        obj_head = tmp_dict["ner"][0][1][0]
        obj_tail = tmp_dict["ner"][0][1][1] + 1

        entity1 = " ".join(tmp_dict["sentences"][0][sub_head:sub_tail])
        entity1_type = tmp_dict["ner"][0][0][2]
        entity2 = " ".join(tmp_dict["sentences"][0][obj_head:obj_tail])
        entity2_type = tmp_dict["ner"][0][1][2]

        if tmp_dict["relations"] == [[]]:
            rel = 'NONE'
        else:
            rel = tmp_dict["relations"][0][0][4]
        tmp_knn.append(reltoid[rel])

        if not reasoning or label_other == 1:
            prompt_query = "\nContext: " + string + "\n" + "Question: Given the context, what is the relation between " + entity1 + " and " + entity2 + "?\n" + "Answer:" + idtoprompt[reltoid[rel]] + ".\n"
            #prompt_query = instance(tmp_dict).reference + " is " + idtoprompt[reltoid[rel]] + ".\n\n"
        else:     
            tmp_query = "What are the clues that lead the relation between \"" + entity1 + "\" and \"" + entity2 + "\" to be " + idtoprompt[reltoid[rel]] + " in the sentence \"" + string + "\"?"


            print(tmp_query)
            results, probs = demo.get_multiple_sample_chat(tmp_query)
            print(results)

            #prompt_query = prompt_query + results[0] +"\n"
            prompt_query = "\nContext: " + string + "\n" + "Question: Given the context, what is the relation between " + entity1 + " and " + entity2 + "?\n" + "Reasoning Process: \n" + results[0] + "\n" + "Answer:" + idtoprompt[reltoid[rel]] + ".\n"
        example_prompt += prompt_query
    return example_prompt, tmp_knn, label_other, example_list

def get_binary_select(pred, tmp_dict, demo, knn_list, reltoid, idtoprompt, args):
    test_example = instance(tmp_dict)
    prompt_list = str()
    for example in knn_list:
        knn_example = instance(example)
        if pred == reltoid[knn_example.rel]:
            prompt_list += knn_example.discriminator + idtoprompt[pred] + "?" + knn_example.answer + " yes.\n"
        else:

            prompt_list += knn_example.discriminator + idtoprompt[pred] + "?" + knn_example.answer + " no.\n"

    
    prompt_list += test_example.discriminator + idtoprompt[pred] + "?" + test_example.answer

    while True:
        try:
            results, probs = demo.get_multiple_sample(prompt_list)
            break
        except:
            continue
    
    print(results[0])
    if "no" in results[0]:
        pred = 0
    return pred, math.exp(probs[0]["token_logprobs"][0])

def get_ft_representation(model, features):
    re_array = []
    dataloader = DataLoader(features, batch_size=32, collate_fn=collate_fn, drop_last=False)
    for ib, batch in enumerate(dataloader):
        model.eval()
        inputs = {'input_ids': batch[0].to(args.device),
              'attention_mask': batch[1].to(args.device),
              'ss': batch[3].to(args.device),
              'os': batch[4].to(args.device),
              }
        with torch.no_grad():
            outputs = model(**inputs).to('cpu').detach().numpy().copy()
        re_array.append(outputs)  
    embeds = np.concatenate(re_array, axis=0)
    return embeds



def run(reltoid, idtoprompt, store_path, args):
    if args.model == 'gpt-3.5-turbo-instruct':
        demo = Demo(
            engine=args.model,
            temperature=0,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            best_of=1,
            logprobs=1,
            )
    else:
        demo = Demo(
            engine=args.model,
            temperature=0,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            best_of=1,
            logprobs=True,
            )

    train_list = get_sample(args.example_dataset)
    test_examples = get_sample(args.test_dataset)

    # train_list = [x for y in example_dict.values() for x in y]
    
    # test_examples = [item for sublist in test_dict.values() for item in sublist]
    # test_examples = random.sample(test_examples, args.num_test)

    # prepare re representations
    res = faiss.StandardGpuResources()
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    knn_model = REModelforKnn(args, config).to(0)
    index_flat = faiss.IndexFlatL2(2 * config.hidden_size)
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    processor = Processor(args, tokenizer)
    train_features = processor.read(args.example_dataset)
    test_features = processor.read(args.test_dataset)

    train_embeds = get_ft_representation(knn_model, train_features)
    gpu_index_flat.add(train_embeds)

    test_embeds = get_ft_representation(knn_model, test_features)

    micro_f1 = 0.0
    for run in range(args.num_run):
        labels = []
        preds = []
        num = 0
        whole_knn = []
        whole_prob = []
        whole_prob_on_rel = []
        store_error_reason = {}
        azure_error = []
        for tmp_dict, tmp_feature in zip(test_examples, test_embeds):
            time.sleep(0.5)
            tmp_knn = []

            label_other = 0
            tmp_dict['feature'] = tmp_feature
            example_prompt, tmp_knn, label_other, knn_list = generate_lm_example(gpu_index_flat, tmp_dict, train_list, knn_model, args.k, reltoid, idtoprompt, args.reasoning, demo, args)
            whole_knn.append(tmp_knn)
            num += 1

            labels.append(reltoid[instance(tmp_dict).rel])
            #prompt_list, subject, target = generate_zero_prompt(tmp_dict, query_dict, relation_dict.keys())
            print(example_prompt)
            prompt_list, subject, target = generate_select_auto_prompt(tmp_dict, example_prompt, reltoid, args.reasoning, args)
            print(prompt_list)
            #results, probs = demo.get_multiple_sample(prompt_list)
            #pred, prob_on_rel = get_results_onebyone(demo, prompt_list, target)
            if args.var and label_other == 1:
                pred = 0
                prob_on_rel = 0
                prob = {"NONE": 1}
            else:
                pred, prob_on_rel, prob, error = get_results_select(demo, prompt_list, reltoid, idtoprompt, args.verbalize, args)
                if error:
                    azure_error.append(tmp_dict)
                if args.discriminator and pred != 0:
                    ori_pred = pred
                    pred, prob = get_binary_select(pred, tmp_dict, demo, knn_list, reltoid, idtoprompt, args)
                    if pred != ori_pred:
                        print("work!")

            whole_prob.append(prob)
            whole_prob_on_rel.append(prob_on_rel)
            preds.append(pred)
            f1_result = compute_f1(preds, labels)
            print(f1_result, end="\n")
            
            if preds[-1] != labels[-1]:
                if args.store_error_reason:
                    error_reason = instance(tmp_dict).get_error_reason(preds[-1], tmp_dict, example_prompt, demo, idtoprompt, reltoid, args)
                    store_error_reason[instance(tmp_dict).id] = error_reason
                with open("{}/negtive.txt".format(store_path), "a") as negf:
                    negf.write(prompt_list + "\n")
                    negf.write(str(reltoid) + "\n")
                    negf.write(str(prob_on_rel) + "\n")
                    negf.write("Prediction: " + str(preds[-1]) + "\n")
                    negf.write("Gold: " + str(labels[-1]) + "\n")
                    negf.write(str({'s':tmp_dict['sentences']}))
                    negf.write("\n-----------------\n")
            else:

                if args.store_error_reason:
                    correct_reason = instance(tmp_dict).get_correct_reason(demo, idtoprompt, reltoid, args)
                    store_error_reason[instance(tmp_dict).id] = correct_reason

            with open("{}/results.txt".format(store_path),"a") as negf:
                negf.write(prompt_list + "\n")
                    
                negf.write(str(reltoid) + "\n")
                negf.write(str(prob_on_rel) + "\n")
                negf.write("Prediction: " + str(preds[-1]) + "\n")
                #negf.write(preds[num])
                negf.write("Gold: " + str(labels[-1]) + "\n")
                negf.write(str(f1_result))
                negf.write("\n")
                #negf.write(labels[num])
                negf.write(str({'s':tmp_dict['sentences']}))
                negf.write("\n-----------------\n")
            print("processing:", 100*num/len(test_examples), "%", end="\n")
        print(classification_report(labels, preds, digits=4))
        report = classification_report(labels, preds, digits=4,output_dict=True)
        if args.store_error_reason:
            with open("stored_reason/{}_dev.txt".format(args.task), "w") as f:
                json.dump(store_error_reason, f)
        with open("{}/labels.csv".format(store_path), "w") as f:
            f.write('\n'.join([str(labels)]))
        with open("{}/preds.csv".format(store_path), "w") as f:
            f.write('\n'.join([str(preds)]))
        with open("{}/prob_on_rel.csv".format(store_path), "w") as f:
            f.write('\n'.join([str(x) for x in whole_prob_on_rel]))
        micro_f1 += f1_result["f1"]
        with open("{}/azure_error.csv".format(store_path), "w") as f:
            f.write('\n'.join([str(azure_error)]))
        with open("{}/knn.csv".format(store_path), "w") as f:
            for line in whole_knn:
                f.write('\n'.join([str(line)]))
                f.write("\n")
        df = pd.DataFrame(report).transpose()
        df.to_csv("{}/result_per_rel.csv".format(store_path))
    avg_f1 = micro_f1 / args.num_run
    print("AVG f1:", avg_f1)
    print(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default=None, required=True, choices=["ace05","semeval","scierc"])
    parser.add_argument("--model", default=None, type=str, required=True)
    parser.add_argument("--num_test", type=int, default=100)
    parser.add_argument("--example_dataset", type=str, default=None, required=True)
    parser.add_argument("--test_dataset", type=str, default=None, required=True)
    parser.add_argument("--fixed_example", type=int, default=1)
    parser.add_argument("--fixed_test", type=int,default=1)
    parser.add_argument("--num_per_rel", type=int, default=2)
    parser.add_argument("--num_na", type=int, default=0)
    parser.add_argument("--num_run", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random_label", type=int, default=0)
    parser.add_argument("--reasoning", type=int, default=0)
    parser.add_argument("--use_knn", type=int, default=0)
    parser.add_argument("--is_simcse", type=int, default=0)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--var", type=int, default=0)
    parser.add_argument("--reverse", type=int, default=0)
    parser.add_argument("--verbalize", type=int, default=0)
    parser.add_argument("--entity_info", type=int, default=0)
    parser.add_argument("--structure", type=int, default=0)
    parser.add_argument("--use_ft", action='store_true')
    parser.add_argument("--self_error", type=int, default=0)
    parser.add_argument("--store_error_reason", type=int, default=0)
    parser.add_argument("--discriminator", type=int, default=0)
    parser.add_argument("--name", type=str, default=0)

    parser.add_argument("--data_dir", default="./data/semeval", type=str)
    parser.add_argument("--model_name_or_path", default="roberta-large", type=str)
    parser.add_argument("--input_format", default="entity_marker", type=str,
                        help="in [entity_mask, entity_marker, entity_marker_punct, typed_entity_marker, typed_entity_marker_punct]")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated.")

    tacred_idtoprompt = {tacred_reltoid[k]:k.upper() for k in tacred_reltoid.keys()}
    scierc_idtoprompt = {scierc_reltoid[k]:k.upper() for k in scierc_reltoid.keys()}
    wiki_idtoprompt = {wiki_reltoid[k]:k.upper() for k in wiki_reltoid.keys()}

    args = parser.parse_args()
    if args.is_simcse == 1:
        args.is_simcse = True
    else:
        args.is_simcse = False
    if args.verbalize == 1:
        args.verbalize = True
    else:
        args.verbalize = False

    if args.entity_info == 1:
        args.entity_info = True
    else:
        args.entity_info = False
    if args.reverse == 1:
        args.reverse = True
    else:
        args.reverse = False
    if args.var and args.no_na:
        raise Exception("Sorry, if focus on no NA examples, please turn var into 0")
    if args.var:
        args.var = True
    else:
        args.var = False
    if args.fixed_example and args.use_knn:
        assert False
    if args.fixed_example == 1:
        args.fixed_example = True
    else:
        args.fixed_example = False

    if args.fixed_test == 1:
        args.fixed_test = True
    else:
        args.fixed_test = False

    if args.reasoning == 1:
        args.reasoning = True
    else:
        args.reasoning = False

    if args.random_label == 1:
        args.random_label = True
    else:
        args.random_label = False
    
    args.device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    store_path = "./results/knn"
    if not os.path.exists(store_path):
        os.mkdir(store_path)
    
    random.seed(args.seed)
    if args.task == "semeval":
        Processor = SemevalProcessor
        run(semeval_reltoid,semeval_idtoprompt, store_path, args)
    elif args.task == "ace05":
        Processor = AceProcessor
        run(ace05_reltoid,ace05_idtoprompt, store_path, args)
    elif args.task == "scierc":
        Processor = SciercProcessor
        run(scierc_reltoid, scierc_idtoprompt, store_path, args)


