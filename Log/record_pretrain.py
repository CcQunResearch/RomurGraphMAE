# -*- coding: utf-8 -*-
# @Time    : 2022/5/31 17:49
# @Author  :
# @Email   :
# @File    : record_pretrain.py
# @Software: PyCharm
# @Note    :
import os
import sys
import json
import math

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dirname, '..'))


def get_value(acc_list):
    mean = round(sum(acc_list) / len(acc_list), 3)
    sd = round(math.sqrt(sum([(x - mean) ** 2 for x in acc_list]) / len(acc_list)), 3)
    maxx = max(acc_list)
    return 'test acc: {:.3f}±{:.3f}'.format(mean, sd), 'max acc: {:.3f}'.format(maxx)


if __name__ == '__main__':
    log_dir_path = os.path.join(dirname, '..', 'Log')

    for filename in os.listdir(log_dir_path):
        if filename[-4:] == 'json':
            print(f'【{filename[:-5]}】')
            filepath = os.path.join(log_dir_path, filename)

            log = json.load(open(filepath, 'r', encoding='utf-8'))
            print('dataset:', log['dataset'])
            print('unsup dataset:', log['unsup dataset'])
            print('vector size:', log['vector size'])
            print('unsup train size:', log['unsup train size'])
            print('runs:', log['runs'])
            print('ft runs:', log['ft runs'])

            print('batch size:', log['batch size'])
            print('num heads:', log['num heads'])
            print('num out heads:', log['num out heads'])
            print('num layers:', log['num layers'])
            print('num hidden:', log['num hidden'])
            print('residual:', log['residual'])
            print('in drop:', log['in drop'])
            print('attn drop:', log['attn drop'])
            print('norm:', log['norm'])
            print('negative slope:', log['negative slope'])
            print('activation:', log['activation'])
            print('mask rate:', log['mask rate'])
            print('drop edge rate:', log['drop edge rate'])
            print('replace rate:', log['replace rate'])
            print('concat hidde:', log['concat hidden'])
            print('pooling:', log['pooling'])

            print('encoder:', log['encoder'])
            print('decoder:', log['decoder'])
            print('loss fn:', log['loss fn'])
            print('alpha l:', log['alpha l'])

            print('lr:', log['lr'])
            print('ft lr:', log['ft lr'])
            print('epochs:', log['epochs'])
            print('ft epochs:', log['ft epochs'])
            print('weight decay:', log['weight decay'])

            record = log['record']
            acc_lists = {10: [], 20: [], 40: [], 80: [], 100: [], 200: [], 300: [], 500: [], 10000: []}

            for run_record in record:
                for re in run_record['record']:
                    acc_lists[re['k']].append(re['mean acc'])
            for key in acc_lists.keys():
                acc, max_acc = get_value(acc_lists[key])
                print(f'k: {key}, {acc}, {max_acc}')
            print()

