'''
Created on Jul 18, 2016
k_means.py

K-means is a an alogithm for finding clusters of similar data without
supervision.  We give it the number of clusters we are looking for and
it lumps the samples together.  If does this by finding a centroid, assigning
the closest smaples to the centroid, recomputing the centroid from
the mean of the samples etc. 

It basically uses a Euclidean distance to evaluate whether a sample 
belongs to a centroid.  So it is good for spherical data, but has
trouble with non spherical data.

Unfortunately, the data in the ocr_utils is not spherical so we
get some odd results.

For this program 
    input a bunch of samples from ocr_utils,
     run K means on them and 
     display the results.
     
     Repeat this for k++ means, that places beginning centroids far away from
     each other.
     
     Run an 'elbow plot' that uses the inertia values from each cluster
     versus the number of clusters.  It shows how many cluster we need
     to get the inertia distortion values to stabilize.
     
    Make some montage plots of images so that we can see what images are
    in the clusters.

from Python Machine Learning by Sebastian Raschka under the following license

The MIT License (MIT)

Copyright (c) 2015, 2016 SEBASTIAN RASCHKA (mail@sebastianraschka.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

@author: richard lyman
'''
import numpy as np
import ocr_utils
import matplotlib.pyplot as plt
n=200

chars_to_train = range(48,51)
columnsXY=(9,17)
column_str = 'column_sum{}'.format(list(columnsXY))
skewRange = np.linspace(-0.5,0.5,81)
input_filters_dict = {'m_label': chars_to_train, 'font': 'E13B'}

# output  the character label and the image and column sums
output_feature_list = ['m_label','image',column_str] 

# read the complete image (20x20) = 400 pixels for each character
ds = ocr_utils.read_data(input_filters_dict=input_filters_dict, 
                            output_feature_list=output_feature_list, 
                            random_state=0)
   
y = ds.train.features[0][:n]
X_image = ds.train.features[1][:n]
X = ds.train.features[2][:n]

# put the ASCII equivalent of the unique characters in y into the legend of the plot
legend=[]
for ys in np.unique(y):
    legend.append('{} \'{}\''.format(ys, chr(ys)))
           
ocr_utils.scatter_plot(X=X, 
                  y=y,
                  legend_entries=legend,
                  axis_labels = ['column {} sum'.format(columnsXY[i]) for i in range(len(columnsXY))], 
                  title='k-means cluster E13B sum of columns')

from sklearn.cluster import KMeans
km = KMeans(n_clusters=3,init='random',n_init=10,max_iter=300,tol=1e-04,random_state=0,n_jobs=-1)
y_km = km.fit_predict(X)

legend=[]
for ys in np.unique(y_km):
    legend.append('{}\''.format(ys))

plt.scatter(km.cluster_centers_[:,0], 
            km.cluster_centers_[:,1], 
            s=250, 
            marker='*', 
            c='red', 
            label='centroids')
           
ocr_utils.scatter_plot(X=X, 
                  y=y_km,
                  legend_entries=legend,
                  axis_labels = ['column {} sum'.format(columnsXY[i]) for i in range(len(columnsXY))], 
                  title='column sums k means centroids')

km = KMeans(n_clusters=3,n_init=10,max_iter=300,tol=1e-04,random_state=0,n_jobs=-1)
y_km = km.fit_predict(X)

legend=[]
for ys in np.unique(y_km):
    legend.append('{}\''.format(ys))

plt.scatter(km.cluster_centers_[:,0], 
            km.cluster_centers_[:,1], 
            s=250, 
            marker='*', 
            c='red', 
            label='k++')
           
ocr_utils.scatter_plot(X=X, 
                  y=y_km,
                  legend_entries='',
                  axis_labels = ['column {} sum'.format(columnsXY[i]) for i in range(len(columnsXY))], 
                  title='column sums k++ means centroids')


for i in range(0,km.cluster_centers_.shape[0]):
    image_index2 = np.argwhere(y_km == i)
    x2d = X_image[image_index2].reshape((image_index2.shape[0],ds.train.num_rows, ds.train.num_columns))
    ocr_utils.montage(x2d,title='k++ cluster {}'.format(i))
    
##############################################
# separate the original images by cluster
# print(km.cluster_centers_.shape)

n=30000

chars_to_train = range(48,58)
columnsXY=range(0,20)
column_str = 'column_sum{}'.format(list(columnsXY))
skewRange = np.linspace(-0.5,0.5,81)
input_filters_dict = {'m_label': chars_to_train, 'font': 'E13B'}

# output  the character label and the image and column sums
output_feature_list = ['m_label','image'] 

# read the complete image (20x20) = 400 pixels for each character
ds = ocr_utils.read_data(input_filters_dict=input_filters_dict, 
                            output_feature_list=output_feature_list, 
                            random_state=0)
   
y = ds.train.features[0][:n]
X_image = ds.train.features[1][:n]
# X = ds.train.features[2][:n]

distortions=[]
for i in range(1,30):
    km = KMeans(n_clusters=i,n_init=10,max_iter=300,tol=1e-04,random_state=0,n_jobs=-1)
    y_km = km.fit_predict(X_image)
    distortions.append(km.inertia_)
    
plt.plot(range(1,30), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
title = '2D image elbow distortion'
plt.title(title)
ocr_utils.show_figures(plt, title)
  
  
km = KMeans(n_clusters=8,n_init=10,max_iter=300,tol=1e-04,random_state=0,n_jobs=-1)
y_km = km.fit_predict(X_image)
        
nClusters = km.cluster_centers_.shape[0]
x2d = []
sz = np.zeros((nClusters))

for i in range(0,nClusters):
    image_index2 = np.argwhere(y_km == i)
    x2d.append( X_image[image_index2].reshape((image_index2.shape[0],ds.train.num_rows, ds.train.num_columns)))
    print (i,x2d[i].shape[0])
    sz[i] = image_index2.shape[0]
             
args= np.argsort(sz)[::-1]
print(sz[args])
print(args)
for i in range(0,nClusters):    
    ocr_utils.montage(x2d[args[i]],title='2D image cluster {}'.format(i))
    
    
print ('\n########################### No Errors ####################################')    

