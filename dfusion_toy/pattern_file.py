import sys
sys.path.append('home/smart/dfusion_toy/')
import os
import re
import random

path = '/run/user/1000/gvfs/smb-share:server=149.201.37.169,share=anwar/densfusion_download_dataset/datasets/dataset_toy_airplane/dataset/rgbd'

files_written = 0
data = open('toy_dataset.txt', 'w')
dataset = []
for file in os.listdir(path):
	#print(file)
	#pattern = re.compile(r'jpg')
	#matches = pattern.finditer(file)

	x = file.split('.')
	#print(x)
	if len(x) == 2:
		if x[1] == 'jpg':
			#print(x[0])
			data.write(x[0]+'\n')
			dataset.append(x[0])
			files_written +=1
		else:
			continue
	else:
		continue

data.close()

print('No. of file written:',files_written)
print('length of dataset list = ', len(dataset))
random.shuffle(dataset)

train_len = int(0.75 * len(dataset))
eval_len = int(0.20*len(dataset))
validation_len = len(dataset) - train_len - eval_len

train_set = dataset[:train_len]
eval_set = dataset[train_len:train_len+eval_len]
validation_set = dataset[train_len + eval_len :]

trainfile = open('train.txt','w')
for line in train_set:
	trainfile.write(line+'\n')
trainfile.close()

evalfile = open('eval.txt','w')
for line in eval_set:
	evalfile.write(line + '\n')
evalfile.close()

valfile = open('validation.txt','w')
for line in validation_set:
	valfile.write(line+'\n')
valfile.close()
