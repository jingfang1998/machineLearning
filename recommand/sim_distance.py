# !/usr/bin/python
# -*- coding: UTF-8 -*-

from math import sqrt

critics = {
    'Lisa':{
        'Lady in the water':2.5,
        'Snake on a plane' :3.5
    },
    'Tom':{
        'Lady in the water':3.0,
        'Snake on a plane' :4.0
    },
    'Jerry':{
        'Lady in the water':2.0,
        'Snake on a plane' :3.0
    },
    'WXM':{
        'Lady in the water':3.3,
        'Snake on a plane' :4.2
    },
    'jhz':{
        'Lady in the water':3.9,
        'Snake on a plane' :4.5
    }
}
"""
欧几里得空间法 计算相似度
"""
def sim_distance(p1, p2):
    c = set(p1.keys()) & set(p2.keys())
    if not c:
        return 0
    sum_of_squares = sum([pow(p1.get(sk) - p2.get(sk), 2) for sk in c])
    p = 1 / (1 + sqrt(sum_of_squares))
    return p

if __name__ == "__main__":
    print(sim_distance(critics["Lisa"], critics["Jerry"]))