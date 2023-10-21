import os
from langchain import PromptTemplate, OpenAI
from langchain.chains import LLMChain
from langchain.chains.base import Chain
import json
from tools.UTILS.utils import *
from tools.IR.client import *
from tools.ChatGLM.GLMclinet import *

def syllogism_reasoning_func(fact, llm):
    legal_articles_set = []
    legal_articles_correct_process = []
    for i in range(3):
        if i == 0:
            prompt_articles_list = PromptTemplate(
                input_variables=["fact"],
                template="基本案情：\n{fact}\n请根据基本案情列举出相关的法条，法条数目不超过5条，法条格式为：《法律名称》+第几条+【罪名】+罪名描述。",
            )
            chain_articles_list = LLMChain(llm=llm, prompt=prompt_articles_list)
            legal_articles = chain_articles_list.predict(fact=fact)
        else:
            prompt_articles_list = PromptTemplate(
                input_variables=["fact", "check_reasons"],
                template="基本案情：\n{fact}\n针对当前法条的修改意见：\n{check_reasons}\n请根据基本案情和针对当前法条的修改意见列举出相关的法条，法条数目不超过5条，法条格式为：《法律名称》+第几条+【罪名】+罪名描述。",
            )
            chain_articles_list = LLMChain(llm=llm, prompt=prompt_articles_list)

            legal_articles = chain_articles_list.predict(fact=fact, check_reasons=check_reasons)
        correct_legal_articles = []
        legal_articles0 = legal_articles
        for kw in legal_articles.split('\n'):
            if kw == '':
                continue
            correct_legal_articles += query(kw, 1)["answer"]

        for cla in correct_legal_articles:
            if cla['all'] in legal_articles_set:
                continue
            legal_articles_set.append(cla['all'])
        legal_articles = '\n'.join(legal_articles_set)
        if len(legal_articles) > 1000:
            legal_articles = '\n'.join([i[:100] for i in legal_articles_set])

        prompt_check_article = PromptTemplate(
            input_variables=["fact", "legal_articles"],
            template="基本案情：\n{fact}\n当前检索到的法条：\n{legal_articles}\n请判断当前检索到的法条是否能够概括基本案情中的罪名，请回答是或者否并换行说明原因。",
        )
        chain_check_articles = LLMChain(llm=llm, prompt=prompt_check_article)
        articles_check = chain_check_articles.predict(fact=fact, legal_articles=legal_articles)
        articles_check = articles_check.strip()
        check_result = articles_check[:2]
        check_reasons = articles_check[2:]
        legal_articles_correct_process.append(
            (legal_articles0, legal_articles, articles_check))
        if check_result[0] == '是':
            break
        else:
            pass
    prompt_object = PromptTemplate(
        input_variables=["fact"],
        template="基本案情：\n{fact}\n请根据基本案情，抽取犯罪客体，并给出犯罪客体的分析，分析过程不超过50字。犯罪客体：刑法所保护而为犯罪所侵犯的社会关系。",
    )
    chain_object = LLMChain(llm=llm, prompt=prompt_object)
    object = chain_object.predict(fact=fact)
    prompt_objective = PromptTemplate(
        input_variables=["fact"],
        template="基本案情：\n{fact}\n请根据基本案情，列举犯罪的客观方面。犯罪客观方面包括：\n1.犯罪时间；\n2.犯罪地点；\n3.犯罪行为；\n4.犯罪结果。\n每一项不超过50字。",
    )
    chain_objective = LLMChain(llm=llm, prompt=prompt_objective)
    objective = chain_objective.predict(fact=fact)
    prompt_subject = PromptTemplate(
        input_variables=["fact"],
        template="基本案情：\n{fact}\n请根据基本案情，抽取犯罪主体，并分析犯罪主体的构成要件，分析过程不超过50字。犯罪主体的构成要件包括：是否达到法定年龄、是否是完全行为能力人。",
    )
    chain_subject = LLMChain(llm=llm, prompt=prompt_subject)
    subject = chain_subject.predict(fact=fact)
    prompt_subjective = PromptTemplate(
        input_variables=["fact"],
        template="基本案情：\n{fact}\n请根据基本案情，列举犯罪主观方面，犯罪主观方面包括：\n1.说明是故意犯罪还是过失犯罪；\n2.说明具体的犯罪意图，犯罪意图的说明不超过50字。",
    )
    chain_subjective = LLMChain(llm=llm, prompt=prompt_subjective)
    subjective = chain_subjective.predict(fact=fact)
    prompt_match = PromptTemplate(
        input_variables=["legal_articles", "object", "objective", "subject", "subjective"],
        template="""
        相关法条：
        {legal_articles}
        犯罪要件：
        犯罪客体：{object}
        犯罪客观方面：{objective}
        犯罪主体：{subject}
        犯罪主观方面：{subjective}
        请根据犯罪要件从相关法条中匹配出符合的法条，并根据相关性由大到小依次列举法条，说明和法条对应的匹配要件。
        列举法条的格式为：《法律名称》+第几条+【罪名】+罪名描述。
        """,
    )
    chain_match = LLMChain(llm=llm, prompt=prompt_match)
    match = chain_match.predict(legal_articles=legal_articles,
                                object=object,
                                objective=objective,
                                subject=subject,
                                subjective=subjective)
    prompt_result = PromptTemplate(
        input_variables=["match"],
        template="""
            法条的匹配结果：
            {match}
            请根据法条的匹配结果，列举判决结果，判决结果包括：\n1.罪名；\n2.刑期。
            罪名要严格遵守法条中的罪名，不可以自行生成罪名，列举出所有符合的罪名。
            """,
    )
    chain_result = LLMChain(llm=llm, prompt=prompt_result)
    result = chain_result.predict(match=match)
    result = correct_charge_name(result)
    return legal_articles_correct_process, legal_articles, object, objective, subject, subjective, match, result


if __name__ == '__main__':
    llm = CustomGLM(url=ChatGLM_URL)
    original_path = 'test data path'
    target_path = 'save result path'
    start_idx = 0
    idx = 0
    length = 23913
    error_num = 0
    data_list = []
    with open(original_path, 'r', encoding="utf-8") as f:
        for line in f:
            data_list.append(json.loads(line))
    if not os.path.exists(os.path.join(target_path, 'all')):
        os.makedirs(os.path.join(target_path, 'all'))

    while idx < length:
        data = data_list[start_idx + idx]
        filename = str(idx)+'_test.json'
        fact = data['fact']
        fact = cut_off(fact, 2000)

        legal_articles_correct_process, legal_articles, object, objective, subject, subjective, match, result = syllogism_reasoning_func(
            fact, llm=llm)
        data['chatglm-generation'] = dict()
        data['chatglm-generation']['legal_articles_correct_process'] = []
        for p in legal_articles_correct_process:
            data['chatglm-generation']['legal_articles_correct_process'].append(
                {'ChatGLM-articles': p[0], 'retrieved-articles': p[1], 'ChatGLM-check-reasons': p[2]})
        data['chatglm-generation']['legal_articles'] = legal_articles
        data['chatglm-generation']['object'] = object
        data['chatglm-generation']['objective'] = objective
        data['chatglm-generation']['subject'] = subject
        data['chatglm-generation']['subjective'] = subjective
        data['chatglm-generation']['match'] = match
        data['chatglm-generation']['result'] = result
        with open(os.path.join(os.path.join(target_path, 'all'), filename), 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        if not os.path.exists(os.path.join(target_path, data['charge'][0])):
            os.mkdir(os.path.join(target_path, data['charge'][0]))
        with open(os.path.join(target_path, data['charge'][0], filename), 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        idx += 1
