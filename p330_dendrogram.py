'''
Created on Jul 21, 2016

from Python Machine Learning by Sebastian Raschka

@author: richard lyman
'''


import ocr_utils
import matplotlib.pyplot as plt

##############################################
# separate the original images by cluster
# print(km.cluster_centers_.shape)

n=300

variables = ['X', 'Y', 'Z']
labels = ['ID_0','ID_1','ID_2','ID_3','ID_4']

chars_to_train = range(48,51)
columnsXY=(9,17)
column_str = 'column_sum{}'.format(list(columnsXY))

input_filters_dict = {'m_label': chars_to_train, 'font': 'E13B'}

# output  the character label and the image and column sums
output_feature_list = ['m_label','image', column_str] 

# read the complete image (20x20) = 400 pixels for each character
ds = ocr_utils.read_data(input_filters_dict=input_filters_dict, 
                            output_feature_list=output_feature_list, 
                            random_state=0)
   
y = ds.train.features[0][:n]
X_image = ds.train.features[1][:n]
X = ds.train.features[2][:n]

from scipy.spatial.distance import pdist

row_dist = pdist(X, metric='euclidean')
print(row_dist)

from scipy.cluster.hierarchy import linkage

#method 1 using a condensed matrix
row_clusters = linkage(row_dist, method='complete', metric='euclidean')

print (row_clusters)
#method 2 using raw data
row_clusters = linkage(X, method='complete', metric='euclidean')

print()
print (row_clusters)

from scipy.cluster.hierarchy import dendrogram

# make dendrogram black (part 1/2)
# from scipy.cluster.hierarchy import set_link_color_palette
# set_link_color_palette(['black'])

row_dendr = dendrogram(row_clusters,  p=12, truncate_mode = 'lastp')

plt.tight_layout()
plt.ylabel('Euclidean distance')
#plt.savefig('./figures/dendrogram.png', dpi=300, 
#            bbox_inches='tight')
title = "Dendogram"
plt.title(title)
ocr_utils.show_figures(plt, title)

print ('\n########################### No Errors ####################################')
