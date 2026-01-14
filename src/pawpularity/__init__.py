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

