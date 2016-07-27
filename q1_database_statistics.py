'''
Created on Jul 25, 2016

@author: richard
'''
import ocr_utils
import numpy as np

    
lst = ocr_utils.get_list(columns=('font'))
print ('The number of fonts = {}'.format(len(lst)))

ds = ocr_utils.read_data()

for i in range(ds.train.num_features):
    names = ds.train.feature_names[i]    
    lengths=len(np.unique(ds.train.features[i]))
    shapes=ds.train.features[i].shape
    print("feature name = {}, \n\tnumber of unique values = {}, feature shape = {}".format(names, lengths, shapes))
    print ('\n########################### No Errors ####################################')
    