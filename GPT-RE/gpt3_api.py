from typing import List
import openai
from openai import OpenAI
import os

MY_KEY = ''
openai.api_key = MY_KEY


class Demo(object):
    def __init__(self, engine, temperature, max_tokens, top_p, frequency_penalty, presence_penalty, best_of, logprobs):
        self.engine = engine
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.best_of = best_of
        self.logprobs = logprobs

    def get_multiple_sample(self, prompt_list):
        client = OpenAI(api_key=openai.api_key)
        response = client.completions.create(
            model=self.engine,
            prompt=prompt_list,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            best_of=self.best_of,
            logprobs=self.logprobs,
        )
        results = [choice.text for choice in response.choices]
        probs = [choice.logprobs for choice in response.choices]
        print(probs)
        return results, probs
    
    def get_multiple_sample_chat(self, prompt_list):
        response = openai.chat.completions.create(
            model=self.engine,
            messages=[
                {"role": "user", "content": prompt_list},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            logprobs=self.logprobs
        )
        results = [choice.message.content for choice in response.choices]
        probs = [choice.logprobs for choice in response.choices]
        return results, probs


def run(prompt_list):
    demo = Demo(
        engine="",  # text-davinci-002: best, text-ada-001: lowest price
        temperature=0,  # control randomness: lowring results in less random completion (0 ~ 1.0)
        max_tokens=8,  # max number of tokens to generate (1 ~ 4,000)
        top_p=1,  # control diversity (0 ~ 1.0)
        frequency_penalty=0,  # how to penalize new tokens based on their existing frequency (0 ~ 2.0)
        presence_penalty=0,  # 这个是对于词是否已经出现过的惩罚，文档上说这个值调高可以增大谈论新topic的概率 (0 ~ 2.0)
        best_of=3,  # 这个是说从多少个里选最好的，如果这里是10，就会生成10个然后选最好的，但是这样会更贵(1 ~ 20)
        logprobs=None
    )
    results, probs = demo.get_multiple_sample_chat(prompt_list)
    print(results[0])
    print(probs[0])


if __name__ == '__main__':
    prompt_list = ["""Refer to the passage below and answer the following question with just one entity. 

 Passage: 
Kanados ministro pirmininko atlyginimą sudaro dvi dalys: atlyginimas už parlamento nario pareigas ir atlyginimas už ministro pirmininko pareigas. Nuo 2020 m. Kanados parlamento nario bazinis atlyginimas yra 182 600 Kanados dolerių. Ministras pirmininkas gauna papildomą atlyginimą, lygų parlamento nario atlyginimui, todėl bendra suma siekia 365 200 CAD. Šis darbo užmokestis neapmokestinamas mokesčiais ir yra papildomas prie tokių išmokų kaip automobilio priedas. Ministro Pirmininko atlyginimas nustatomas pagal Kanados parlamento įstatymą. Šį aktą kasmet peržiūri Vidaus ekonomikos taryba, kuri gali rekomenduoti keisti parlamento narių ir Ministro Pirmininko atlyginimus.

Title: Prime Minister of Canada Content: and his or her family. All of the aforementioned is supplied by the Queen-in-Council through budgets approved by parliament, as is the prime minister's total annual compensation of CAD$347,400. The Prime Minister's total compensation consists of the Member of the House of Commons Basic Sessional Indemnity of CAD$172,400, the Prime Minister Salary of CAD$172,400, and the Prime Minister Car Allowance of CAD$2000. Should a serving or former prime minister die, he or she is accorded a state funeral, wherein their casket lies in state in the Centre Block of Parliament Hill. Only Bowell and the Viscount Bennett were given private 

 Question: what is the prime minister of canada salary? 

 The answer is"""][0]
    run(prompt_list)
