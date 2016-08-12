'''
Created on Jul 18, 2016
silhouette.py

A silhouette plot shows how well the samples are bound to a single
centroid selected by k-means and how well they are separated from the
other clusters.

Typically the cohesion and dissimilarity coefficients that make up
the silhouette are calculated using Euclidean distance.

This program shows silhouette plot using a small number of clusters.

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
n=1000

chars_to_train = (48,50)
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

from matplotlib import cm
from sklearn.metrics import silhouette_samples
from sklearn.cluster import KMeans

km = KMeans(n_clusters=2, 
            init='k-means++', 
            n_init=10, 
            max_iter=300,
            tol=1e-04,
            random_state=0)
y_km = km.fit_predict(X)

cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(i / n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, 
            edgecolor='none', color=color)

    yticks.append((y_ax_lower + y_ax_upper) / 2)
    y_ax_lower += len(c_silhouette_vals)
    
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color="red", linestyle="--") 

plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')
title = 'Silhouettes'
plt.title(title)
plt.tight_layout()
ocr_utils.show_figures(plt, title)
print ('\n########################### No Errors ####################################')
