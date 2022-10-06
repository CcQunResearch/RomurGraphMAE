# -*- coding: utf-8 -*-
# @Time    : 2022/9/28 19:32
# @Author  :
# @Email   :
# @File    : record_sup.py
# @Software: PyCharm
# @Note    :
import os
import sys
import json
import math
import numpy as np

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dirname, '..'))

cal_mean = -10

if __name__ == '__main__':
    log_dir_path = os.path.join(dirname, '..', 'Log')

    for filename in os.listdir(log_dir_path):
        if filename[-4:] == 'json':
            print(f'【{filename[:-5]}】')
            filepath = os.path.join(log_dir_path, filename)

            log = json.load(open(filepath, 'r', encoding='utf-8'))
            print('dataset:', log['dataset'])
            print('vector_size:', log['vector size'])
            print('unsup train size:', log['unsup train size'])
            print('runs:', log['runs'])

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
            print('epochs:', log['epochs'])
            print('weight decay:', log['weight decay'])

            print('use unlabel:', log['use unlabel'])
            print('use unsup loss:', log['use unsup loss'])

            acc_list = []
            for run in log['record']:
                # mean_acc = run['mean acc']
                mean_acc = round(np.mean(run['test accs'][cal_mean:]), 3)
                acc_list.append(mean_acc)

            mean = round(sum(acc_list) / len(acc_list), 3)
            sd = round(math.sqrt(sum([(x - mean) ** 2 for x in acc_list]) / len(acc_list)), 3)
            maxx = max(acc_list)
            print('test acc: {:.3f}±{:.3f}'.format(mean, sd))
            print('max acc: {:.3f}'.format(maxx))
            print()
