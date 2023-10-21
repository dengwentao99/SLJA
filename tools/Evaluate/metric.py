import math
import re
from utils import *
from numpy import *

def precision_recall_fscore_k2(y_true, y_pred, num=3):
    if not isinstance(y_true[0], list):
        y_true = [[each] for each in y_true]
    if not isinstance(y_pred[0], list):
        y_pred = [[each] for each in y_pred]
    y_pred = [each[0:num] for each in y_pred]
    unique_label = count_unique_label(y_true, y_pred)
    res = {}
    result = ''
    R = []
    for each in unique_label:
        cur_res = []
        tp_fn = 0
        for i in y_true:
            if each in i:
                tp_fn += 1
        tp_fp = 0
        for i in y_pred:
            if each in i:
                tp_fp += 1
        tp = 0
        for i in range(len(y_true)):
            if each in y_true[i] and each in y_pred[i]:
                tp += 1
        support = tp_fn
        try:
            precision = tp/tp_fp
        except ZeroDivisionError:
            precision = 0
        try:
            recall = tp/tp_fn
        except ZeroDivisionError:
            recall = 0
        try:
            f1_score = 2/((1/precision)+(1/recall))
        except ZeroDivisionError:
            f1_score = 0
        cur_res.append(precision)
        cur_res.append(recall)
        cur_res.append(f1_score)
        cur_res.append(support)
        res[str(each)] = cur_res
        R.append(recall)
    title = '\t'+'precision@'+str(num)+'\t'+'recall@'+str(num)+'\t'+'f1_score@'+str(num)+'\t'+'support'+'\n'
    result += title
    res_dict = dict()
    for k, v in sorted(res.items()):
        cur = str(k)+'\t'+str(v[0])+'\t'+str(v[1])+'\t'+str(v[2])+'\t'+str(v[3])+'\n'
        res_dict[k] = {'precision': v[0], 'recall': v[1], 'f1-score': v[2], 'support': v[3]}
        result += cur
    true_label = set()
    for i in y_true:
        for j in i:
            true_label.add(j)
    sums = len(true_label)
    weight_info = [(v[0] * 1, v[1] * 1, v[2] * 1) for k, v in res.items()]
    weight_precision = 0
    weight_recall = 0
    weight_f1_score = 0
    for ul, each in zip(unique_label, weight_info):
        weight_precision += each[0]
        weight_recall += each[1]
        weight_f1_score += each[2]
    weight_precision /= sums
    weight_recall /= sums
    weight_f1_score /= sums
    last_line = 'avg_total' + '\t' + str(round(weight_precision, 4)) + '\t' + str(round(weight_recall, 4)) + '\t' + str(
        round(weight_f1_score, 4)) + '\t' + str(sums)
    res_dict['macro avg'] = last_line.split('\t')
    result += last_line
    return result, res_dict


def count_unique_label(y_true, y_pred):
    unique_label = []
    for each in y_true:
        for j in each:
            if j not in unique_label:
                unique_label.append(j)
    for i in y_pred:
        for j in i:
            if j not in unique_label:
                unique_label.append(j)
    unique_label = list(set(unique_label))
    return unique_label


def extract_terms(sentence):
    sentence = cn_num2an_num(sentence)
    patterns = [
        r'\d+年[零]?\d+个月以上\d+年[零]?\d+个月以下有期徒刑[；，、或者]{0,2}[拘役]{0,2}[；，、或者]{0,2}[管制]{0,2}',
        r'\d+年[零]?\d+个月以上\d+年以下有期徒刑[；，、或者]{0,2}[拘役]{0,2}[；，、或者]{0,2}[管制]{0,2}',
        r'\d+年以上\d+年[零]?\d+个月以下有期徒刑[；，、或者]{0,2}[拘役]{0,2}[；，、或者]{0,2}[管制]{0,2}',
        r'\d+个月以上\d+年[零]?\d+个月以下有期徒刑[；，、或者]{0,2}[拘役]{0,2}[；，、或者]{0,2}[管制]{0,2}',
        r'\d+年以上\d+年以下有期徒刑[；，、或者]{0,2}[拘役]{0,2}[；，、或者]{0,2}[管制]{0,2}',
        r'\d+个月以上\d+年以下有期徒刑[；，、或者]{0,2}[拘役]{0,2}[；，、或者]{0,2}[管制]{0,2}',
        r'\d+个月以上\d+个月以下有期徒刑[；，、或者]{0,2}[拘役]{0,2}[；，、或者]{0,2}[管制]{0,2}',
        r'\d+年以上有期徒刑[；，、或者]{0,2}[无期徒刑]{0,4}[；，、或者]{0,2}[死刑]{0,2}',
        r'\d+年以上有期徒刑[；，、或者]{0,2}[拘役]{0,2}[；，、或者]{0,2}[管制]{0,2}',
        r'\d+至\d+年有期徒刑[；，、或者]{0,2}[拘役]{0,2}[；，、或者]{0,2}[管制]{0,2}',
        r'\d+年[零]?\d+个月以上有期徒刑[；，、或者]{0,2}[无期徒刑]{0,4}[；，、或者]{0,2}[死刑]{0,2}',
        r'\d+年[零]?\d+个月以下有期徒刑[；，、或者]{0,2}[拘役]{0,2}[；，、或者]{0,2}[管制]{0,2}',
        r'\d+年以下有期徒刑[；，、或者]{0,2}[拘役]{0,2}[；，、或者]{0,2}[管制]{0,2}[；，、或者]{0,2}[拘役]{0,2}[；，、或者]{0,2}[管制]{0,2}',
        r'\d+个月以下有期徒刑[；，、或者]{0,2}[拘役]{0,2}[；，、或者]{0,2}[管制]{0,2}',
        r'\d+个月以上有期徒刑[；，、或者]{0,2}[无期徒刑]{0,4}[；，、或者]{0,2}[死刑]{0,2}',
        r'\d+年[零]?\d+个月至\d+年[零]?\d+个月有期徒刑',
        r'\d+年[零]?\d+个月至\d+年有期徒刑',
        r'\d+年至\d+年[零]?\d+个月有期徒刑',
        r'\d+个月至\d+年\d+个月有期徒刑',
        r'\d+年至\d+年有期徒刑',
        r'\d+个月至\d+年[有期徒刑]?',
        r'\d+个月至\d+个月有期徒刑',
        r'\d+\-\d+年有期徒刑',
        r'\d+个月有期徒刑',
        r'\d+年有期徒刑',
        r'有期徒刑\d+年\d+个月',
        r'有期徒刑\d+年',
        r'有期徒刑\d+年以上\d+年以下',
        r'有期徒刑\d+年以下',
        r'不超过\d+年',
        r'不超过\d+年\d+个月',
        r'拘役[；，、或者]{0,2}[管制]{0,2}',
        r'管制',
        r'死刑',
        r'无期徒刑',
    ]
    pattern = '|'.join(patterns)
    res = re.findall(pattern, sentence)
    return res


def term_metric5(target, predict, num=3):
    assert len(target) == len(predict)
    tp = [0, 0, 0, 0, 0]
    tp_fn = [0, 0, 0, 0, 0]
    tp_fp = [0, 0, 0, 0, 0]
    for t, p in zip(target, predict):
        class_predict = []
        class_target = []

        for p_ in p[: num]:
            if p_[0] != [0, 1e+100]:
                class_predict.append(0)
                break
        if t[0] > 0:
            class_target.append(0)
        for p_ in p[: num]:
            if p_[1] != [0, 1e+100]:
                class_predict.append(1)
                break
        if t[1] > 0:
            class_target.append(1)
        leng = 235
        for p_ in p[: num]:
            if p_[2] != [0, 1e+100] and p_[2][0] <= t[2] <= p_[2][1]:
                class_predict.append(2)
                leng = min(leng, p_[2][1] - p_[2][0] + 1)
                break
        if t[2] > 0:
            class_target.append(2)
        for p_ in p[: num]:
            if p_[3] == True:
                class_predict.append(3)
                break
        if t[3] == True:
            class_target.append(3)
        for p_ in p[: num]:
            if p_[4] == True:
                class_predict.append(4)
                break

        penalty = max([1, math.log(leng, 12)])
        if t[4] == True:
            class_target.append(4)
        if len(set(class_predict) & set(class_target)) == 0:
            a0 = 0
        elif 2 in set(class_predict) & set(class_target):
            a0 = (len(set(class_predict) & set(class_target)) - 1 + 1 / penalty)
        elif 2 not in set(class_predict) & set(class_target):
            a0 = len(set(class_predict) & set(class_target))

        if 2 in set(class_predict):
            a1 = (len(set(class_predict)) - 1 + 1 / penalty)
        elif 2 not in set(class_predict):
            a1 = len(set(class_predict))

        if 2 in set(class_target):
            b1 = (len(set(class_target)) - 1 + 1 / 1)
        elif 2 not in set(class_target):
            b1 = len(set(class_target))

        if 0 in set(class_target):
            tp[0] += a0
            tp_fp[0] += a1
            tp_fn[0] += b1
        if 1 in set(class_target):
            tp[1] += a0
            tp_fp[1] += a1
            tp_fn[1] += b1
        if 2 in set(class_target):
            tp[2] += a0
            tp_fp[2] += a1
            tp_fn[2] += b1
        if 3 in set(class_target):
            tp[3] += a0
            tp_fp[3] += a1
            tp_fn[3] += b1
        if 4 in set(class_target):
            tp[4] += a0
            tp_fp[4] += a1
            tp_fn[4] += b1

    P = [tp[i] / (tp_fp[i] + 1e-12) for i in range(5)]
    R = [(0 if tp_fn[i] == 0 else tp[i] / tp_fn[i]) for i in range(5)]
    F = [0 if P[i] + R[i] == 0 else 2 * (P[i] * R[i]) / (P[i] + R[i]) for i in range(5)]
    return round(sum(P) / len(P), 4), \
        round(sum(R) / len(R), 4), \
        round(sum(F) / len(F), 4)
