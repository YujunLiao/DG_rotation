import os
import numpy
import re
from openpyxl import Workbook
import math

wb = Workbook()
# grab the active worksheet
ws = wb.active

# path = "/home/lyj//Files/project/pycharm/DG_rotation/trainer_utils/output_manager/output_file/"
file_name = 'DA_rotation/resnet50/0.6_0.6_0.7_0.1/'
path = "/home/lyj/Files/project/pycharm/DG_rotation/trainer_utils/output_manager/output_file/era/"+ file_name

files= os.listdir(path)
files.sort()
# print(files)
column=0
# files = ['cartoon_art_painting','art_painting_cartoon',  'art_painting_sketch', 'art_painting_photo', ]
for file in files: #遍历文件夹
    if file=='original_record':
        continue
    file_words = file.split('_')
    domain_info = file_words[0][0] + file_words[1][0]
    ws[chr(ord('A') + column) + str(1)] = domain_info
    print(domain_info, file_words)
    file_read = open(path+file, 'r')
    row = 1
    for line in file_read:
        words = line.split(' ')
        if words[0]=='Accuracy':
            row += 1
            ws[chr(ord('A') + column) + str(row)] = words[11].split(':')[1][:-1]
    column += 1
print(file_name)
wb.save("./result.xlsx")