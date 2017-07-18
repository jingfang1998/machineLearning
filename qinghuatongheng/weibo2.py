# !/usr/bin/python
# -*- coding: UTF-8 -*-
import pandas as pd
def weibo():
    data = pd.read_csv("data/weibo.csv")

    print(data["text"])
if __name__ == "__main__":
    weibo()