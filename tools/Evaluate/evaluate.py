import os

from utils import *
import re
from sklearn.metrics import classification_report
from metric import *
import jieba
from nltk.translate.bleu_score import sentence_bleu
from rouge_chinese import Rouge

with open('valid_charge_list.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    valid_charges = [i.strip() for i in lines]


def evaluate(data_name_list, data_list, prefix=''):
    charge_correct = 0
    total_num = 0
    charge_predict_list = []

    charge_target_list = []
    article_target_list = []
    term_target_list = []

    charge_predict_all_list = []
    article_predict_all_list = []
    term_predict_all_list = []

    for data_name, data in zip(data_name_list, data_list):
        charge_groundtruth = data['charge']
        charge_groundtruth = [i for i in charge_groundtruth if i in valid_charges]
        term_groundtruth = (data['term_of_imprisonment']['guanzhi'], data['term_of_imprisonment']['juyi'],
                            data['term_of_imprisonment']['imprisonment'], data['term_of_imprisonment']['death_penalty'],
                            data['term_of_imprisonment']['life_imprisonment'])

        itm = prefix
        chatgpt_answer = correct_charge_name(data['chatgpt-generation' + itm]['result'])

        article_groundtruth = []
        for i in data['laws']:
            if 0 <= int(cn_num2an_num(i[1])) <= 600 and i[0]['title'] == '中华人民共和国刑法':
                if cn_num2an_num(i[1]) not in article_groundtruth:
                    article_groundtruth.append(cn_num2an_num(i[1]))
        article_groundtruth = article_groundtruth[:1]
        if len(article_groundtruth) > 1 and article_groundtruth[1] >= 102:
            article_groundtruth.append(article_groundtruth[1])
        if 'match' in data['chatgpt-generation' + itm]:
            chatgpt_articles = data['chatgpt-generation' + itm]['match']
        else:
            chatgpt_articles = data['chatgpt-generation' + itm]['legal_articles']
        charge_chatgpt = chatgpt_answer.strip()
        pattern = '|'.join(valid_charges)
        charge_chatgpt = re.findall(pattern, charge_chatgpt)
        charge_chatgpt_tmp = []
        for cc in charge_chatgpt:
            if cc not in charge_chatgpt_tmp:
                charge_chatgpt_tmp.append(cc)
        charge_chatgpt = charge_chatgpt_tmp
        terms_chatgpt = extract_terms(chatgpt_answer)
        terms_chatgpt_ = []
        for i in terms_chatgpt:
            if i not in terms_chatgpt_:
                terms_chatgpt_.append(i)
        terms_chatgpt_list = []
        for i in terms_chatgpt_:
            a = extract_scope(i)
            terms_chatgpt_list.append(a)

        article_chatgpt = []
        try:
            chatgpt_articles = cn_num2an_num(chatgpt_articles)
            chatgpt_articles = chatgpt_articles.strip().replace('罪、', '罪，').replace('罪；', '罪，').replace('罪和', '罪，')
            pattern1 = '第(\d+)条'
            pattern2 = '刑法(\d+)条'
            a = re.findall(pattern1, chatgpt_articles) + re.findall(pattern2, chatgpt_articles)
            for i in a:
                if cn_num2an_num(i) not in article_chatgpt and 0 <= int(cn_num2an_num(i)) <= 700:
                    article_chatgpt.append(cn_num2an_num(i))
        except:
            pass
        total_num += 1
        charge_predict_all_list.append(charge_chatgpt)
        article_predict_all_list.append(article_chatgpt)
        term_predict_all_list.append(terms_chatgpt_list)
        charge_target_list.append(charge_groundtruth)
        article_target_list.append(article_groundtruth)
        term_target_list.append(term_groundtruth)
        if len(charge_predict_list) != len(charge_target_list):
            charge_predict_list.append('None')
        if charge_groundtruth[0] in charge_chatgpt:
            charge_correct += 1
        else:
            charge_correct += 0

    assert len(charge_target_list) == len(charge_predict_list) == len(term_target_list) == len(term_predict_all_list)
    for num in [1, 3, 5]:
        res = precision_recall_fscore_k2(charge_target_list, charge_predict_all_list, num=num)
        print('charge: R', num, res[1]['macro avg'][2])
        P, R, F = term_metric5(term_target_list, term_predict_all_list, num=num)
        print('term: R', num, R)
        res = precision_recall_fscore_k2(article_target_list, article_predict_all_list, num=num)
        print('article: R', num, res[1]['macro avg'][2])


charge_subject_dict = dict()
with open('charge_intent.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for i in lines:
        t = i.strip().split('\t')
        charge_subject_dict[t[1]] = t[0]

def evaluate_subjective(data_name_list, data_list, prefix=''):
    predict_subjectives = []
    target_subjectives = []
    tot = 0
    for data_name, data in zip(data_name_list, data_list):
        subjective = data['chatgpt-generation']['subjective']
        subjective = subjective.replace('\n', '')
        subjective = subjective.replace('\r', '')
        subjective_pat_list = [
            '主观方面为.*?[，。]',
            '在事故中存在.*?[，。]',
            '犯罪意图是.*?[，。]',
            '有.*?的主观意图[，。]',
            '有犯罪的主观方面.*?[，。,]',
            '构成.*?[，。]',
            '其.*?犯罪[，。]',
            '其有.*?的主观.*?[，。,]',
            '具有.*?的主观.*?[，。]',
            '具有主观上的.*?[，。,]',
            '具有明显的.*?[，。]',
            '出于.*?[，。]',
            '有.*?的意图[，。,]',
            '有.*?的犯罪意图[，。,]',
            '犯罪行为是出于.*?[，。]',
            '具有.*?意图[，。,]',
            '具有.*?的.*?意图[，。,]',
            '存在.*?的意图[，。,]',
            '存在.*?的犯罪意图[，。,]',
            '有明显的.*?[，。,]',
            '有犯罪意图.*?[，。,]',
            '具有.*?的主观方面[，。]',
            '犯罪主观方面.*?[，。]',
            '出于.*?的意图[，。]',
            '具有明显的.*?犯罪的主观意图[，。]',
            '具有犯罪.*?[，。]',
            '属于.*?[，。]',
            '主观方面是.*?[，。]',
            '犯罪主观方面为.*?[，。]',
            '犯罪主观方面：.*?[，。]',
            '存在.*?主观方面[，。]',
            '存在.*?主观意图[，。]',
            '存在主观上的.*?[，。,]',
            '具有.*?主观方面[，。]',
            '具有.*?[，。]',
            '犯.*?犯罪[，。]',
            '主观上具有.*?[，。]',
            '其.*?犯罪意图明显[，。]',
            '系.*?犯罪[，。]',
            '被告人.*?[，。]',
            '犯罪意图为.*?[，。]',
            '.*?故意.*?[，]',
            '故意犯罪',
            '犯罪故意'
        ]
        pattern_subjective_1 = '|'.join(subjective_pat_list)
        subjective_1 = re.findall(pattern_subjective_1, subjective)
        if len(subjective_1) == 0:
            subjective = ''
        elif '故意' in subjective_1[0]:
            subjective = '故意'
        elif '过失' in subjective_1[0]:
            subjective = '过失'
        else:
            subjective = ''
        predict_subjectives.append(subjective)
        target_charge = data['charge']

        for i in target_charge[:1]:
            target_subjectives.append(charge_subject_dict[i])
    result = classification_report(target_subjectives, predict_subjectives, digits=4, output_dict=True)
    print((result['故意']['precision'] + result['过失']['precision']) / 2,
          (result['故意']['precision'] + result['过失']['recall']) / 2,
          (result['故意']['precision'] + result['过失']['f1-score']) / 2)


def evaluate_subject(data_name_list, predict_data_list, groundtruth_data_list, prefix=''):
    correct_num = 0
    for data_name, predict_data, groundtruth_data in zip(data_name_list, predict_data_list, groundtruth_data_list):
        predict_subject = predict_data['chatgpt-generation' + prefix]['subject']
        predict_subject = predict_subject.replace('\n', '')
        predict_subject = predict_subject.replace('\r', '')
        groundtruth_subject = groundtruth_data['syllogism']['minor_premise']['subject']
        groundtruth_subject = groundtruth_subject.replace('\n', '')
        groundtruth_subject = groundtruth_subject.replace('\r', '')
        subject_pat_list = [
            '犯罪主体是(.*?)[，。,]',
            '犯罪主体为(.*?)[，。,]',
            '犯罪主体：(.*?)构成要件',
            '犯罪主体：(.*?)[，。,]',
            '犯罪主体包括(.*?)[，。,]',
            '犯罪主体包括(.*?)[，。,]',
            '构成要件:(.*?)已达',
            '(.*?)具有',
            '犯罪主体是(.*?)[，。,]',
        ]
        pattern_subject = '|'.join(subject_pat_list)
        predict_subject_l = re.findall(pattern_subject, predict_subject)
        if len(predict_subject_l) == 0:
            predict_subject_f = 'none'
        else:
            for k in predict_subject_l[0]:
                if len(k) != 0:
                    predict_subject_f = k
                    break
            else:
                predict_subject_f = 'none'
        groundtruth_subject_l = re.findall(pattern_subject, groundtruth_subject)
        if len(groundtruth_subject_l) == 0:
            groundtruth_subject_f = 'none'
        else:
            for k in groundtruth_subject_l[0]:
                if len(k) != 0:
                    groundtruth_subject_f = k
                    break
            else:
                groundtruth_subject_f = 'none'
        predict_subject_f = predict_subject_f.replace('被告人', '').replace('。', '').replace('犯罪主体', '').replace('的', '').replace('等人', '')
        groundtruth_subject_f = groundtruth_subject_f.replace('被告人', '')
        if predict_subject_f == groundtruth_subject_f:
            correct_num += 1
    print(correct_num / len(data_name_list))


def evaluate_object(data_name_list, predict_data_list, groundtruth_data_list, prefix=''):
    tot = 0
    correct_num = 0
    for data_name, predict_data, groundtruth_data in zip(data_name_list, predict_data_list, groundtruth_data_list):
        predict_object = predict_data['chatgpt-generation' + prefix]['object']
        predict_object = predict_object.replace('\n', '')
        predict_object = predict_object.replace('\r', '')
        groundtruth_object = groundtruth_data['syllogism']['minor_premise']['object']
        groundtruth_object = groundtruth_object.replace('\n', '')
        groundtruth_object = groundtruth_object.replace('\r', '')
        object_pat_list = [
            '犯罪客体是(.*?)[，。,]',
            '犯罪客体为(.*?)[，。,]',
            '犯罪客体包括(.*?)[，。,]',
            '犯罪客体为(.*?)$',
            '犯罪客体(.*?)[，。,]',
        ]
        pattern_object = '|'.join(object_pat_list)
        object_pat_list2 = [
            '侵犯了(.*?)[，。,]',
            '侵害了(.*?)[，。,]',
            ',犯罪客体是(.*?)[，。,]',
            '即(.*?)[，。,]',
            '犯罪客体为(.*?)[，。,]',
            '犯罪客体为(.*?)$',
            '犯罪客体：(.*?)$',
            '犯罪客体为(.*?)$',
            '犯罪客体分析：(.*?)$',
        ]
        pattern_object2 = '|'.join(object_pat_list2)
        predict_object_l = re.findall(pattern_object2, predict_object)
        if len(predict_object_l) == 0:
            predict_object_f = 'none'
        else:
            for k in predict_object_l[0]:
                if len(k) != 0:
                    predict_object_f = k
                    break
            else:
                predict_object_f = 'none'
        predict_object_f = predict_object_f.replace('被害人', '')
        groundtruth_object_l = re.findall(pattern_object, groundtruth_object)
        if len(groundtruth_object_l) == 0:
            groundtruth_object_f = 'none'
        else:
            for k in groundtruth_object_l[0]:
                if len(k) != 0:
                    groundtruth_object_f = k
                    break
            else:
                groundtruth_object_f = 'none'
        predict_object_f = predict_object_f.replace('被告人', '').replace('。', '').replace('犯罪客体', '').replace('的', '').replace('等人', '')
        groundtruth_object_f = groundtruth_object_f.replace('被害人', '').replace('。', '').replace('的', '')

        if (groundtruth_object_f in predict_object_f):
            correct_num += 1
    print(correct_num / len(data_name_list))

def calculate_bleu_rouge(reference, candidate):
    reference_jieba = jieba.lcut(reference)
    candidate_jieba = jieba.lcut(candidate)
    bleu_score = sentence_bleu([reference_jieba], candidate_jieba, weights=[0.25, 0.25, 0.25, 0.25])
    rouge = Rouge()
    rouge_score = rouge.get_scores(hyps=[' '.join(reference_jieba)], refs=[' '.join(candidate_jieba)])
    return bleu_score, rouge_score


def evaluate_objective(data_name_list, data_list, final_path, type):
    bleu_scores = []
    rouge_scores = []
    labeled_list = os.listdir(final_path)
    labeled_ids = [i.split('_')[0] for i in labeled_list]
    for data_name, data in zip(data_name_list, data_list):
        data_id = data_name.split('_')[0]
        if data_id not in labeled_ids:
            continue
        idx = labeled_ids.index(data_id)
        with open(os.path.join(final_path, labeled_list[idx]), 'r', encoding='utf-8') as f_r:
            reference_data = json.load(f_r)
        object = data['chatgpt-generation']['object'].replace('\n', '')
        subject = data['chatgpt-generation']['subject'].replace('\n', '')
        objective = data['chatgpt-generation']['objective'].replace('\n', '')
        subjective = data['chatgpt-generation']['subjective'].replace('\n', '')
        match = data['chatgpt-generation']['match'].replace('\n', '')

        object_reference = reference_data['syllogism']['minor_premise']['object'].replace('\n', '')
        subject_reference = reference_data['syllogism']['minor_premise']['subject'].replace('\n', '')
        objective_reference = reference_data['syllogism']['minor_premise']['objective'].replace('\n', '')
        subjective_reference = reference_data['syllogism']['minor_premise']['subjective'].replace('\n', '')

        match_reference = reference_data['syllogism']['conclusion']['applicable_article'].replace('\n', '')
        if type == 'subject':
            bleu_score, rouge_score = calculate_bleu_rouge(subject_reference, subject)
            rouge_scores.append(rouge_score[0]["rouge-l"]['f'])
        if type == 'subjective':
            bleu_score, rouge_score = calculate_bleu_rouge(subjective_reference, subjective)
            rouge_scores.append(rouge_score[0]["rouge-l"]['f'])
        if type == 'object':
            bleu_score, rouge_score = calculate_bleu_rouge(object_reference, object)
            rouge_scores.append(rouge_score[0]["rouge-l"]['f'])
        if type == 'objective':
            bleu_score, rouge_score = calculate_bleu_rouge(objective_reference, objective)
            rouge_scores.append(rouge_score[0]["rouge-l"]['f'])
        if type == 'match':
            bleu_score, rouge_score = calculate_bleu_rouge(match_reference, match)
            rouge_scores.append(rouge_score[0]["rouge-l"]['f'])
    print('ROUGE', sum(rouge_scores) / len(rouge_scores))


if __name__ == '__main__':
    output_folder = r'save result path'
    data_name_list = list_file_names(output_folder)
    data_list = []
    for data_name in data_name_list:
        with open(os.path.join(output_folder, data_name), 'r', encoding='utf-8') as f:
            data_list.append(json.load(f))
    evaluate(data_name_list, data_list, prefix='')


    groundtruth_folder = r'label path'
    groundtruth_files = os.listdir(groundtruth_folder)
    predict_files = [i.split('-')[0] for i in groundtruth_files]
    groundtruth_data_list = []
    predict_data_list = []
    for data_name in groundtruth_files:
        with open(os.path.join(groundtruth_folder, data_name), 'r', encoding='utf-8') as f:
            groundtruth_data_list.append(json.load(f))

    for data_name in predict_files:
        with open(os.path.join(output_folder, data_name), 'r', encoding='utf-8') as f:
            predict_data_list.append(json.load(f))
    evaluate_object(predict_files, predict_data_list, groundtruth_data_list)
    evaluate_objective(predict_files, predict_data_list, type='subject', final_path=groundtruth_folder)
    evaluate_objective(predict_files, predict_data_list, type='object', final_path=groundtruth_folder)
    evaluate_objective(predict_files, predict_data_list, type='objective', final_path=groundtruth_folder)
    evaluate_subject(predict_files, predict_data_list, groundtruth_data_list)
    evaluate_subjective(predict_files, predict_data_list)


