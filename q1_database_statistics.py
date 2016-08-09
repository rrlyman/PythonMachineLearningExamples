'''
Created on Jul 25, 2016



@author: richard
'''
import ocr_utils
import numpy as np

# # read and show the character images for each font variant
# # output only the character label and the image
# fl = ['m_label','image'] 
# fd = {'font': 'AGENCY', 'fontVariant': 'AGENCY FB', 'm_label':(73,)}
# ds = ocr_utils.read_data(input_filters_dict=fd, output_feature_list=fl, dtype=np.int32)   
# y,X = ds.train.features
# X2D = np.reshape(X, (X.shape[0], ds.train.num_rows, ds.train.num_columns ))
# title = '{}: {}'.format('AGENCY','AGENCY Is')
# ocr_utils.montage(X2D, title=title)
        
lst = ocr_utils.get_list(input_filters_dict = {'font':()})

print('\n\nAvailable fonts:')
import pprint
pp = pprint.PrettyPrinter()
pp.pprint(lst)
# 
# for font in lst:
#     input_filters_dict = {'font':font, 'm_label': range(100)}    
#     ds = ocr_utils.read_data(input_filters_dict=input_filters_dict)
# 
#     for i in range(ds.train.num_features):
#         names = ds.train.feature_names[i]    
#         lengths=len(np.unique(ds.train.features[i]))
#         shapes=ds.train.features[i].shape
#         print("\n\nfeature name = {}, \n\tnumber of unique values = {}, feature shape = {}".format(names, lengths, shapes))


#############################################################################
# read and show the character images for each font variant
# output only the character label and the image
fl = ['m_label','image'] 
for font in lst:    
    lst2 = ocr_utils.get_list(input_filters_dict={'font':font, 'fontVariant':()})
    for f,fontVariant in lst2:
        fd = {'font': font, 'fontVariant': fontVariant}
        ds = ocr_utils.read_data(input_filters_dict=fd, output_feature_list=fl, dtype=np.int32)   
        y,X = ds.train.features
        X2D = np.reshape(X, (X.shape[0], ds.train.num_rows, ds.train.num_columns ))
        title = '{}: {}'.format(font,fontVariant)
        ocr_utils.show_examples(X2D, y, title=title)
    
print ('\n########################### No Errors ####################################')
    