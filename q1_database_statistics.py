'''
Created on Jul 25, 2016



@author: richard
'''
import ocr_utils
import numpy as np 
        
df1 = ocr_utils.get_list(input_filters_dict = {'font':()})
unique_fonts=[]
unique_fontVariants=[]
unique_m_labels=[]
unique_strengths=[]
unique_italics=[]
unique_orientations=[]

#############################################################################
# read and show the character images for each font variant
# output only the character label and the image

for font in df1:    
    df2 = ocr_utils.get_list(input_filters_dict = {'font':font,'fontVariant':(), 'm_label':(),'strength':(),'italic':(),'orientation':()})
    unique_fonts = np.unique( np.append(unique_fonts, df2['font']))
    u1= np.unique(df2['fontVariant'])    
    unique_fontVariants = np.unique(np.append(unique_fontVariants, u1))    
    u2 = np.unique(df2['m_label'])
    unique_m_labels = np.unique(np.append(unique_m_labels,u2))   
    u3 = np.unique(df2['strength'])
    unique_strengths =  np.unique(np.append(unique_strengths,u3))
    u4 = np.unique(df2['italic'])
    unique_italics = np.unique(np.append(unique_italics,u4))
    u5 =np.unique( df2['orientation'])
    unique_orientations = np.unique(np.append(unique_orientations,u5))
    print ('\n{}, fontVariants={}, labels = {}, strengths = {}, italics = {}, orientations = {}\n'.format(font[0], len(u1), 
                                                                                                               len(u2), len(u3),                                                                                                              len(u4), len(u5))) 
    for fontVariant in u1:
        fd = {'font': font, 'fontVariant': fontVariant}
        ds = ocr_utils.read_data(input_filters_dict=fd, output_feature_list=['m_label','image'] , dtype=np.int32)   
        y,X = ds.train.features
        X2D = np.reshape(X, (X.shape[0], ds.train.num_rows, ds.train.num_columns ))
        title = '{}: {}'.format(font,fontVariant)
        ocr_utils.show_examples(X2D, y, title=title)
       
print ('unique fonts={}, fontVariants={}, labels = {}, strengths = {}, italics = {}, orientations = {}'.format(len(unique_fonts), len(unique_fontVariants), 
                                                                                                               len(unique_m_labels), len(unique_strengths), 
                                                                                                               len(unique_italics), len(unique_orientations)))
    
    
print ('\n########################### No Errors ####################################')
    