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
