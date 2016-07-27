#!/usr/bin/python
'''simple_e13b_display.py displays a plot of the characters in the E13B font

See the explanation of the E13B character set in ocr_utils.load_E13B. 

Created on Jun 20, 2016

from Python Machine Learning by Sebastian Raschka

@author: richard lyman

'''
import ocr_utils
import numpy as np

#############################################################################
# read images and scatter plot

# retrieve 400 sets of target numbers and column sums
#    y: the ascii characters 48 and 49 ('0', '1')
#    X: the sum of the vertical pixels in the rows in horizontal columns 9 and 17
ascii_characters_to_train = (48,49)
columnsXY = (9,17)       
y, X, y_test,  X_test, labels  = ocr_utils.load_E13B(chars_to_train=ascii_characters_to_train , columns=columnsXY,nChars=256) 

# put the ASCII equivalent of the unique characters in y into the legend of the plot
legend=[]
for ys in np.unique(y):
    legend.append('{} \'{}\''.format(ys, chr(ys)))
           
ocr_utils.scatter_plot(X=X, 
                  y=y,
                  legend_entries=legend,
                  axis_labels = ['column {} sum'.format(columnsXY[i]) for i in range(len(columnsXY))], 
                  title='E13B sum of columns')

#############################################################################
# read and show character images for '0', and '1'
# select the digits in columnsXY in the E13B font

fd = {'m_label': ascii_characters_to_train, 'font': 'E13B'}

# output only the character label and the image
fl = ['m_label','image'] 

# read the complete image (20x20) = 400 pixels for each character
ds = ocr_utils.read_data(input_filters_dict=fd, output_feature_list=fl, dtype=np.int32)   
y,X = ds.train.features

# change to a 2D shape 
X=np.reshape(X,(X.shape[0],ds.train.num_rows, ds.train.num_columns))
ocr_utils.montage(X,title='E13B Characters {}'.format(legend))

#############################################################################
# read and show the character images of digits 0 through 9 

ascii_characters_to_train = range(48,58)
fd = {'m_label': ascii_characters_to_train, 'font': 'E13B'}

ds = ocr_utils.read_data(input_filters_dict=fd, output_feature_list=fl, dtype=np.int32)   
y,X = ds.train.features

num_characters = len(np.unique(y))

# display 10 of each image
zz = np.zeros((10*num_characters, ds.train.num_rows* ds.train.num_columns) )
for i,ys  in enumerate(np.unique(y)):
    x = X[y==ys]    
    n = min(10,x.shape[0])
    zz[i*10:i*10+n,:] = x[0:n,:]

# change to a 2D shape and plot 
zz=np.reshape(zz,(zz.shape[0],ds.train.num_rows, ds.train.num_columns))
ocr_utils.montage(zz, title='Sample of E13B Characters')

print ('\n########################### No Errors ####################################')
