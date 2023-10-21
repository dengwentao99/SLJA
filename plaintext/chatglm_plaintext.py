import os
from langchain import PromptTemplate, OpenAI
from langchain.chains import LLMChain
from langchain.chains.base import Chain
from tools.IR.Ir import *
import json
from UTILS.utils import *
from tools.IR.client import *
from tools.ChatGLM.GLMclinet import *

def syllogism_reasoning_func(fact, llm):
    prompt_article = PromptTemplate(
        input_variables=["fact"],
        template="""
        基本案情：
        {fact}
        请根据基本案情列举出相关的法条，法条数目不超过5条。
        法条格式为：《法律名称》+第几条+【罪名】+罪名描述。
        """,
    )
    chain_article = LLMChain(llm=llm, prompt=prompt_article)
    legal_articles = chain_article.predict(fact=fact)
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
            请根据基本案情列举判决结果，判决结果包括：\n1.罪名；\n2.刑期。
            """,
    )
    chain_result = LLMChain(llm=llm, prompt=prompt_result)
    result = chain_result.predict(fact=fact)
    result = correct_charge_name(result)
    return legal_articles, result


if __name__ == '__main__':
    llm = CustomGLM(url="ChatGLM URL")
    original_path = 'original data path'
    target_path = 'save result path'
    start_idx =0
    idx = 0
    length = 23913
    error_num = 0
    data_list = []
    with open(original_path, 'r', encoding="utf-8") as f:
        for line in f:
            data_list.append(json.loads(line))
    if not os.path.exists(os.path.join(target_path, 'all')):
        os.makedirs(os.path.join(target_path, 'all'))
    current_data_list = os.listdir(os.path.join(target_path, 'all'))

    while idx < length:
        data = data_list[start_idx + idx]
        filename = str(data['id'])+'_test.json'
        fact = data['fact']
        fact = cut_off(fact, 2000)
        legal_articles, result = syllogism_reasoning_func(fact, llm=llm)
        data['chatglm-generation'] = dict()
        data['chatglm-generation']['legal_articles'] = legal_articles
        data['chatglm-generation']['result'] = result
        with open(os.path.join(os.path.join(target_path, 'all'), filename), 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        if not os.path.exists(target_path+data['charge'][0]):
            os.mkdir(os.path.join(target_path, data['charge'][0]))
        with open(os.path.join(target_path, data['charge'][0], filename), 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        idx += 1

