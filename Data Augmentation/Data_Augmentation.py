
# coding: utf-8

# # AUGMENTOR
# 
# This code file is responsible for Data Augmentation. We have more than 6000 samples for Elliptical Galaxy and 5000 samples for Spiral Galaxy. However, for Irregular Galaxy, we have an image dataset of only 11 samples. So, here we apply the Augmentor module to use those 11 samples for creating more than 1200 samples for the Irregular Galaxy type. 

# In[1]:

#Importing relevant modules
import Augmentor
import warnings
warnings.filterwarnings('ignore')


# In[2]:

#Setting up the Augmentor Pipeline
p = Augmentor.Pipeline("C:/Users/Diganta/Desktop/Courses_and_Projects/Projects/Bennet/irregular")


# In[3]:

#Defining our method of Image Augmentation
p.rotate90(probability=0.5)
p.rotate270(probability=0.5)
p.flip_left_right(probability=0.8)
p.flip_top_bottom(probability=0.3)
p.crop_random(probability=1, percentage_area=0.5)
p.resize(probability=1.0, width=120, height=120)


# In[4]:

#Producing a sample of 1300 Augmented Irregular Galaxy type image samples from a set of 11 input images
p.sample(1300)


# In[ ]:



