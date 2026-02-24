
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt
import os
import PIL
from PIL import UnidentifiedImageError
from sklearn.model_selection import StratifiedShuffleSplit
pd.options.mode.chained_assignment = None

num_rows = 162
num_cols = 2
data = [[None] * num_cols] * num_rows 
newdf = pd.DataFrame(data, columns = ['Id', 'Weight'])

data_path = '/Users/wmeikle/Downloads/petfinder-pawpularity-score/'
data = pd.read_csv(data_path+'train.csv')

sssplit = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
for train_index, test_index in sssplit.split(data, data['Pawpularity']):
    training_set = data.iloc[train_index]
    eval_set = data.iloc[test_index]
    
training_set['Pawpularity'].hist(label='Training set')
eval_set['Pawpularity'].hist(label='Eval set')
plt.title('Pawpularity score distribution in training and test set')
plt.xlabel('Pawpularity score')
plt.ylabel('Count')
plt.legend(loc='upper right')
plt.show()


training_set['Id'] = training_set['Id'].apply(lambda x: '/Users/wmeikle/Downloads/petfinder-pawpularity-score/train/'+x+'.jpg')
training_set[['Id', 'Pawpularity']].to_csv('training_set.csv', header=False, index=False)
eval_set['Id'] = eval_set['Id'].apply(lambda x: '/Users/wmeikle/Downloads/petfinder-pawpularity-score/train/'+x+'.jpg')
eval_set[['Id', 'Pawpularity']].to_csv('eval_set.csv', header=False, index=False)

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3

path = '/Users/wmeikle/Downloads/petfinder-pawpularity-score/train/'
training_img = os.listdir(path)
rand_idx = np.random.randint(0, len(training_img)-1)
rand_img = training_img[rand_idx]

show_image(path+rand_img)

