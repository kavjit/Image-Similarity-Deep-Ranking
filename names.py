#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 14:25:23 2018

@author: kavjit
"""

import os
import pandas as pd
from random import randint, choice


rootDir = "tiny-imagenet-200/train/"
file_table = pd.DataFrame(columns = ['classid','class','image'])
fileSet = []
i = 0
img_class = 0
for dir_, _, files in os.walk(rootDir):
    for fileName in files:
        relDir = os.path.relpath(dir_, rootDir)
        relFile = os.path.join(relDir, fileName)
        if '.JPEG' in relFile:
            if i%500 == 0:
                img_class+=1
                print(img_class)
            file_table.loc[i,'image'] = relFile
            file_table.loc[i, 'class'] = relFile[0:9]
            file_table.loc[i,'classid'] = img_class
            i+=1


file_table.columns = ['classid','class','query']
file_table['inclass'] = 0
file_table['outclass'] = 0

for index,row in file_table.iterrows():
    print(index)
    current_class = row[0]
    gen_start = (current_class-1)*500
    incls_index = index
    while incls_index == index:
        incls_index = randint(gen_start,gen_start+499) #for inclass image
    
    if current_class == 1:
        outcls_index = randint(gen_start+500,99999)
    elif current_class == 200:
        outcls_index = randint(0,gen_start-1)
    else:
        first_seg = randint(0,gen_start-1)
        second_seg = randint(gen_start+500,99999)
        outcls_index = choice([first_seg,second_seg]) 
    
    #print('{}, {}'.format(incls_index,outcls_index))
    file_table.loc[index,'inclass'] = file_table.loc[incls_index,'query']
    file_table.loc[index,'outclass'] = file_table.loc[outcls_index,'query']
    


file_table.to_csv('file_table.csv',index = False)








