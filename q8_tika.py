#!/usr/bin/python
'''experiment with apache tika


@author: richard lyman
'''
import tika
tika.initVM()
from tika import parser
parsed = parser.from_file('15-01-01 459_Mont_Lyman.jpg') 
#parsed = parser.from_file('img20150901_15233271bw.jpg')
print (parsed["metadata"])
print (parsed["content"])
print ('\n########################### No Errors ####################################')