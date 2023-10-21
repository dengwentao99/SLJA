import json
import os
import cn2an
import re


def read_json(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    return data_list


def list_file_names(folder_path):
    filenames = os.listdir(folder_path)
    filenames = [i for i in filenames if i[-2:]!='py']
    return filenames


def cut_off(passage, length_threshold):
    utterances = passage.split('。')
    result_passage = []
    total_length = 0
    for utterance in utterances:
        if len(utterance) + 1 + total_length <= length_threshold:
            result_passage.append(utterance)
            total_length = len(utterance) + 1 + total_length
        else:
            break
    return '。'.join(result_passage)


def chinese2unicode(sentence):
    return sentence.encode('unicode_escape')


def cn_num2an_num(sentence):
    return cn2an.transform(sentence, "cn2an")


def extract_scope(sentence):
    # 管制, 拘役, 有期徒刑, 死刑, 无期徒刑
    terms = [[0, 1e100], [0, 1e100], [0, 1e100], False, False]
    sentence = cn2an.transform(sentence, 'cn2an')

    if '死刑' in sentence:
        terms[3] = True
    if '无期徒刑' in sentence:
        terms[4] = True
    if '管制' in sentence:
        terms[0][0] = 3
        terms[0][1] = 2 * 12
    if '拘役' in sentence:
        terms[1][0] = 1
        terms[1][1] = 6
    if '有期徒刑' in sentence:
        terms[2][0] = 6
        terms[2][1] = 20 * 12


    m = re.findall(r'(\d+)年[零]?(\d+)个月以上(\d+)年[零]?(\d+)个月以下有期徒刑', sentence)
    if len(m) > 0 and int(m[0][0]) * 12 + int(m[0][1]) >= terms[2][0] and int(m[0][2]) * 12 + int(m[0][3]) <= terms[2][
        1]:
        terms[2][0] = int(m[0][0]) * 12 + int(m[0][1])
        terms[2][1] = int(m[0][2]) * 12 + int(m[0][3])
        return terms
    m = re.findall(r'(\d+)年[零]?(\d+)个月以上(\d+)年以下有期徒刑', sentence)
    if len(m) > 0 and int(m[0][0]) * 12 + int(m[0][1]) >= terms[2][0] and int(m[0][2]) * 12 <= terms[2][1]:
        terms[2][0] = int(m[0][0]) * 12 + int(m[0][1])
        terms[2][1] = int(m[0][2]) * 12
        return terms
    m = re.findall(r'(\d+)年以上(\d+)年[零]?(\d+)个月以下有期徒刑', sentence)
    if len(m) > 0 and int(m[0][0]) >= terms[2][0] and int(m[0][1]) * 12 + int(m[0][2]) <= terms[2][1]:
        terms[2][0] = int(m[0][0]) * 12
        terms[2][1] = int(m[0][1]) * 12 + int(m[0][2])
        return terms
    m = re.findall(r'(\d+)个月以上(\d+)年[零]?(\d+)个月以下有期徒刑', sentence)
    if len(m) > 0 and int(m[0][0]) >= terms[2][0] and int(m[0][1]) * 12 + int(m[0][2]) <= terms[2][1]:
        terms[2][0] = int(m[0][0])
        terms[2][1] = int(m[0][1]) * 12 + int(m[0][2])
        return terms

    m = re.findall(r'(\d+)年以上(\d+)年以下有期徒刑', sentence)
    if len(m) > 0 and int(m[0][0]) * 12 >= terms[2][0] and int(m[0][1]) * 12 <= terms[2][1]:
        terms[2][0] = int(m[0][0]) * 12
        terms[2][1] = int(m[0][1]) * 12
        return terms
    m = re.findall(r'(\d+)个月以上(\d+)年以下有期徒刑', sentence)
    if len(m) > 0 and int(m[0][0]) >= terms[2][0] and int(m[0][1]) * 12 <= terms[2][1]:
        terms[2][0] = int(m[0][0])
        terms[2][1] = int(m[0][1]) * 12
        return terms
    m = re.findall(r'(\d+)个月以上(\d+)个月以下有期徒刑', sentence)
    if len(m) > 0 and int(m[0][0]) >= terms[2][0] and int(m[0][1]) <= terms[2][1]:
        terms[2][0] = int(m[0][0])
        terms[2][1] = int(m[0][1])
        return terms
    m = re.findall(r'(\d+)至(\d+)年有期徒刑', sentence)
    if len(m) > 0 and int(m[0][0]) * 12 >= terms[2][0] and int(m[0][1]) * 12 <= terms[2][1]:
        terms[2][0] = int(m[0][0]) * 12
        terms[2][1] = int(m[0][1]) * 12
        return terms

    m = re.findall(r'(\d+)年[零]?(\d+)个月以上有期徒刑', sentence)
    if len(m) > 0 and int(m[0][0]) * 12 + int(m[0][1]) >= terms[2][0]:
        terms[2][0] = int(m[0][0]) * 12 + int(m[0][1])
        terms[2][1] = 20 * 12
        return terms
    m = re.findall(r'(\d+)年[零]?(\d+)个月以下有期徒刑', sentence)
    if len(m) > 0 and int(m[0][0]) * 12 + int(m[0][1]) <= terms[2][1]:
        terms[2][0] = 6
        terms[2][1] = int(m[0][0]) * 12 + int(m[0][1])
        return terms

    m = re.findall(r'(\d+)年以下有期徒刑', sentence)
    if len(m) > 0 and int(m[0]) * 12 <= terms[2][1]:
        terms[2][0] = 6
        terms[2][1] = int(m[0]) * 12
        return terms
    m = re.findall(r'(\d+)个月以下有期徒刑', sentence)
    if len(m) > 0 and int(m[0]) <= terms[2][1]:
        terms[2][0] = 6
        terms[2][1] = int(m[0])
        return terms
    m = re.findall(r'(\d+)年以上有期徒刑', sentence)
    if len(m) > 0 and int(m[0]) * 12 >= terms[2][0]:
        terms[2][0] = int(m[0]) * 12
        terms[2][1] = 20 * 12
        return terms
    m = re.findall(r'(\d+)个月以上有期徒刑', sentence)
    if len(m) > 0 and int(m[0]) >= terms[2][0]:
        terms[2][0] = int(m[0])
        terms[2][1] = 20 * 12
        return terms

    m = re.findall(r'(\d+)年[零]?(\d+)个月至(\d+)年[零]?(\d+)个月有期徒刑', sentence)
    if len(m) > 0 and int(m[0][0]) * 12 + int(m[0][1]) >= terms[2][0] and int(m[0][2]) * 12 + int(m[0][3]) <= terms[2][
        1]:
        terms[2][0] = int(m[0][0]) * 12 + int(m[0][1])
        terms[2][1] = int(m[0][2]) * 12 + int(m[0][3])
        return terms
    m = re.findall(r'(\d+)年[零]?(\d+)个月至(\d+)年有期徒刑', sentence)
    if len(m) > 0 and int(m[0][0]) * 12 + int(m[0][1]) >= terms[2][0] and int(m[0][2]) * 12 <= terms[2][1]:
        terms[2][0] = int(m[0][0]) * 12 + int(m[0][1])
        terms[2][1] = int(m[0][2]) * 12
        return terms
    m = re.findall(r'(\d+)年至(\d+)年[零]?(\d+)个月有期徒刑', sentence)
    if len(m) > 0 and int(m[0][0]) >= terms[2][0] and int(m[0][1]) * 12 + int(m[0][2]) <= terms[2][1]:
        terms[2][0] = int(m[0][0]) * 12
        terms[2][1] = int(m[0][1]) * 12 + int(m[0][2])
        return terms
    m = re.findall(r'(\d+)个月至(\d+)年(\d+)个月有期徒刑', sentence)
    if len(m) > 0 and int(m[0][0]) >= terms[2][0] and int(m[0][1]) * 12 + int(m[0][2]) <= terms[2][1]:
        terms[2][0] = int(m[0][0])
        terms[2][1] = int(m[0][1]) * 12 + int(m[0][2])
        return terms

    m = re.findall(r'(\d+)年至(\d+)年有期徒刑', sentence)
    if len(m) > 0 and int(m[0][0]) * 12 >= terms[2][0] and int(m[0][1]) * 12 <= terms[2][1]:
        terms[2][0] = int(m[0][0]) * 12
        terms[2][1] = int(m[0][1]) * 12
        return terms
    m = re.findall(r'(\d+)个月至(\d+)年[有期徒刑]?', sentence)
    if len(m) > 0 and int(m[0][0]) >= terms[2][0] and int(m[0][1]) * 12 <= terms[2][1]:
        terms[2][0] = int(m[0][0])
        terms[2][1] = int(m[0][1]) * 12
        return terms
    m = re.findall(r'(\d+)个月至(\d+)个月有期徒刑', sentence)
    if len(m) > 0 and int(m[0][0]) >= terms[2][0] and int(m[0][1]) <= terms[2][1]:
        terms[2][0] = int(m[0][0])
        terms[2][1] = int(m[0][1])
        return terms

    m = re.findall(r'(\d+)[个.*月]*至(\d+)个.*月.*拘役', sentence)
    if len(m) > 0 and int(m[0][0]) >= terms[2][0] and int(m[0][1]) <= terms[2][1]:
        terms[1][0] = int(m[0][0])
        terms[1][1] = int(m[0][1])
        return terms
    m = re.findall(r'(\d+)-(\d+)个.*月.*拘役', sentence)
    if len(m) > 0 and int(m[0][0]) >= terms[2][0] and int(m[0][1]) <= terms[2][1]:
        terms[1][0] = int(m[0][0])
        terms[1][1] = int(m[0][1])
        return terms
    m = re.findall(r'(\d+)到(\d+)个.*月.*拘役', sentence)
    if len(m) > 0 and int(m[0][0]) >= terms[2][0] and int(m[0][1]) <= terms[2][1]:
        terms[1][0] = int(m[0][0])
        terms[1][1] = int(m[0][1])
        return terms
    m = re.findall(r'拘役(\d+)-(\d+)个.*月.*', sentence)
    if len(m) > 0 and int(m[0][0]) >= terms[2][0] and int(m[0][1]) <= terms[2][1]:
        terms[1][0] = int(m[0][0])
        terms[1][1] = int(m[0][1])
        return terms
    m = re.findall(r'拘役(\d+)到(\d+)个.*月.*', sentence)
    if len(m) > 0 and int(m[0][0]) >= terms[2][0] and int(m[0][1]) <= terms[2][1]:
        terms[1][0] = int(m[0][0])
        terms[1][1] = int(m[0][1])
        return terms
    m = re.findall(r'(\d+)个.*月以上(\d+)个.*月以下.*拘役', sentence)
    if len(m) > 0 and int(m[0][0]) >= terms[2][0] and int(m[0][1]) <= terms[2][1]:
        terms[1][0] = int(m[0][0])
        terms[1][1] = int(m[0][1])
        return terms

    m = re.findall(r'拘役(\d+)[个.*月]*至(\d+)个.*月', sentence)
    if len(m) > 0 and int(m[0][0]) >= terms[2][0] and int(m[0][1]) <= terms[2][1]:
        terms[1][0] = int(m[0][0])
        terms[1][1] = int(m[0][1])
        return terms

    m = re.findall(r'拘役(\d+)个.*月以上(\d+)个.*月以下', sentence)
    if len(m) > 0 and int(m[0][0]) >= terms[2][0] and int(m[0][1]) <= terms[2][1]:
        terms[1][0] = int(m[0][0])
        terms[1][1] = int(m[0][1])
        return terms

    m = re.findall(r'拘役(\d+)个.*月', sentence)
    if len(m) > 0 and int(m[0]) >= terms[2][0] and int(m[0]) <= terms[2][1]:
        terms[1][0] = int(m[0])
        terms[1][1] = int(m[0])
        return terms

    m = re.findall(r'(\d+)个.*月拘役', sentence)
    if len(m) > 0 and int(m[0]) >= terms[2][0] and int(m[0]) <= terms[2][1]:
        terms[1][0] = int(m[0])
        terms[1][1] = int(m[0])
        return terms

    m = re.findall(r'(\d+)\-(\d+)年有期徒刑', sentence)
    if len(m) > 0 and int(m[0][0]) * 12 >= terms[2][0] and int(m[0][1]) * 12 <= terms[2][1]:
        terms[2][0] = int(m[0][0]) * 12
        terms[2][1] = int(m[0][1]) * 12
        return terms

    m = re.findall(r'(\d+)年(\d+)个月有期徒刑', sentence)
    if len(m) > 0 and 6 >= terms[2][0] and int(m[0][0]) * 12 + int(m[0][1]) <= terms[2][1]:
        terms[2][0] = int(m[0][0]) * 12 + int(m[0][1])
        terms[2][1] = int(m[0][0]) * 12 + int(m[0][1])
        return terms

    m = re.findall(r'(\d+)年有期徒刑', sentence)
    if len(m) > 0 and 6 >= terms[2][0] and int(m[0]) * 12 <= terms[2][1]:
        terms[2][0] = int(m[0]) * 12
        terms[2][1] = int(m[0]) * 12
        return terms

    m = re.findall(r'有期徒刑(\d+)年(\d+)个月', sentence)
    if len(m) > 0 and 6 >= terms[2][0] and int(m[0][0]) * 12 + int(m[0][1]) <= terms[2][1]:
        terms[2][0] = int(m[0][0]) * 12 + int(m[0][1])
        terms[2][1] = int(m[0][0]) * 12 + int(m[0][1])
        return terms

    m = re.findall(r'有期徒刑(\d+)年', sentence)
    if len(m) > 0 and 6 >= terms[2][0] and int(m[0]) * 12 <= terms[2][1]:
        terms[2][0] = int(m[0]) * 12
        terms[2][1] = int(m[0]) * 12
        return terms

    m = re.findall(r'(\d+)个月有期徒刑', sentence)
    if len(m) > 0 and 6 and int(m[0]) <= terms[2][1]:
        terms[2][0] = int(m[0])
        terms[2][1] = int(m[0])
        return terms
    return terms


def correct_charge_name(sentence):
    return sentence.replace('罪名列表：', '罪名：').replace('罪名如下：', '罪名：')
