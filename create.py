'''
author: meng-zha
data: 2020/06/01
'''
import os
from tqdm import tqdm
import numpy as np
from fire import Fire

def create(root_path):
    train_split = []
    val_split = []
    test_split = []

    for root,dirs,files in os.walk(os.path.join(root_path,'train')):
        for name in tqdm(files):
            if 'jpg' in name:
                rand = np.random.uniform(0,1)
                image = name.split('.')[0]
                if rand<0.8:
                    train_split.append(image+'\n')
                else:
                    val_split.append(image+'\n')

    for root,dirs,files in os.walk(os.path.join(root_path,'val')):
        for name in tqdm(files):
            if 'jpg' in name:
                image = name.split('.')[0]
                test_split.append(image+'\n')

    if not os.path.exists('./imagesets'):
        os.mkdir('./imagesets')
    with open('./imagesets/train_split.txt','w') as f:
        f.writelines(train_split)
    with open('./imagesets/val_split.txt','w') as f:
        f.writelines(val_split)
    with open('./imagesets/test_split.txt','w') as f:
        f.writelines(test_split)

if __name__ == "__main__":
    Fire(create)