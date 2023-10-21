# 根据基本案情直接生成法条，罪名和刑期

import os
from langchain import PromptTemplate, OpenAI
from langchain.chains import LLMChain
from langchain.chains.base import Chain
from tools.IR.Ir import *
import json
from tools.UTILS.utils import *
from tools.IR.client import *

def syllogism_reasoning_func(fact):

    os.environ["OPENAI_API_KEY"] = "OpenAI Key"
    model_names = ["text-davinci-002", "text-davinci-003", "gpt-3.5-turbo"]
    llm = OpenAI(model_name=model_names[0], temperature=0)
    prompt_article = PromptTemplate(
        input_variables=["fact"],
        template="""
        基本案情：{fact}\n请根据基本案情列举出符合的法条，法条数目不超过5条，法条格式为：《法律名称》+第几条+【罪名】+罪名描述。法条的格式样例：《中华人民共和国刑法》第一条 【立法宗旨】为了惩罚犯罪，保护人民，根据宪法，结合我国同犯罪作斗争的具体 经验及实际情况，制定本法。
        """,
    )
    chain_article = LLMChain(llm=llm, prompt=prompt_article)
    legal_articles = chain_article.predict(fact=fact)
    legal_articles0 = legal_articles
    legal_articles_set = []
    correct_legal_articles = []
    for kw in legal_articles.split('\n'):
        if kw == '':
            continue
        correct_legal_articles += query(kw, '1')["answer"]
    for cla in correct_legal_articles:
        if cla['all'] in legal_articles_set:
            continue
        legal_articles_set.append(cla['all'])
    legal_articles = '\n'.join(legal_articles_set)
    if len(legal_articles) > 1000:
        legal_articles = '\n'.join([i[:100] for i in legal_articles_set])
    prompt_result = PromptTemplate(
        input_variables=["fact"],
        template="""
            基本案情：
            {fact}
            请根据基本案情给出裁判结果，裁判结果包括罪名和刑期。
            裁判结果的格式为：罪名：盗窃罪\n 刑期：三年以上五年以下有期徒刑。
            """,
    )
    chain_result = LLMChain(llm=llm, prompt=prompt_result)
    result = chain_result.predict(fact=fact)
    result = correct_charge_name(result)
    return legal_articles, result, legal_articles0


if __name__ == '__main__':
    original_path = 'test data path'
    target_path = 'save result path'
    if not os.path.exists(os.path.join(target_path, 'all')):
        os.makedirs(os.path.join(target_path, 'all'))
    start_idx = 0
    idx = 0
    length = 23913
    error_num = 0
    data_list = []
    with open(original_path, 'r', encoding="utf-8") as f:
        for line in f:
            data_list.append(json.loads(line))
    with open('valid_charge_list.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    while idx < length:
        current_data_list = os.listdir(os.path.join(target_path, 'all'))
        data = data_list[start_idx + idx]
        filename = str(data['id'])+'_test.json'
        if filename in current_data_list:
            idx += 1
            continue
        fact = data['fact']
        fact = cut_off(fact, 2000)
        legal_articles, result, legal_articles0 = syllogism_reasoning_func(fact)
        data['chatgpt-generation'] = dict()
        data['chatgpt-generation']['ChatGPT-articles'] = legal_articles0
        data['chatgpt-generation']['legal_articles'] = legal_articles
        data['chatgpt-generation']['result'] = result

        with open(os.path.join(os.path.join(target_path, 'all'), filename), 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        if not os.path.exists(target_path+data['charge'][0]):
            os.mkdir(os.path.join(target_path, data['charge'][0]))
        with open(os.path.join(target_path, data['charge'][0], filename), 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        idx += 1


