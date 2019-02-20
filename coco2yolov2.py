
# coding: utf-8

# In[1]:


import os
import numpy as np
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import cv2


# In[2]:


train_annFile='./annotations/instances_train2017.json'
val_annFile='./annotations/instances_val2017.json'


# In[3]:


train_coco=COCO(train_annFile)


# In[4]:


categories=['person','bicycle','car','motorcycle','truck','bus']
catIds_lst =train_coco.getCatIds(catNms=categories)


# In[5]:


imgIds_lst=list()
for i in catIds_lst:    
    imgIds_lst += train_coco.getImgIds(catIds=i)
imgIds_lst=list(set(imgIds_lst))


# In[6]:


image=cv2.imread('train2017/'+'0'*(12-len(str(imgIds_lst[75])))+str(imgIds_lst[75])+'.jpg')
plt.axis('off')
plt.imshow(image)
plt.show()


# In[7]:


for i in imgIds_lst[75:76]:
    img=train_coco.loadImgs(i)[0]
    annIds = train_coco.getAnnIds(imgIds=img['id'], catIds=catIds_lst, iscrowd=None)
    anns = train_coco.loadAnns(annIds)
    path='train2017/'+'0'*(12-len(str(img['id'])))+str(img['id'])+'.txt'
    with open(path,'w') as f:
        for ann in anns:
            if type(ann['bbox']) == list:
                xmin=ann['bbox'][0]
                ymin=ann['bbox'][1]
                width=ann['bbox'][2]
                height=ann['bbox'][3]
                x=int(xmin+(width/2))/img['width']
                y=int(ymin+(height/2))/img['height']
                r_width=width/img['width']
                r_height=height/img['height']
                upper=(int((x*img['width'])-((r_width*img['width'])/2)),int((y*img['height'])-((r_height*img['height'])/2)))
                lower=(int((x*img['width'])+((r_width*img['width'])/2)),int((y*img['height'])+((r_height*img['height'])/2)))                
                cv2.rectangle(image, upper, lower ,(128,128,128), thickness = 4)
                f.write(str(ann['category_id']) + ' ' +str(round(x,3)) + ' ' +str(round(y,3)) +' ' + str(round(r_width,3)) + ' ' +str(round(r_height,3)))
                f.write('\n')


# In[8]:


plt.axis('off')
plt.imshow(image)
plt.show()


# In[ ]:


for i in imgIds_lst:
    img=train_coco.loadImgs(i)[0]
    annIds = train_coco.getAnnIds(imgIds=img['id'], catIds=catIds_lst, iscrowd=None)
    anns = train_coco.loadAnns(annIds)
    path='train2017/'+'0'*(12-len(str(img['id'])))+str(img['id'])+'.txt'
    with open(path,'w') as f:
        for ann in anns:
            if type(ann['bbox']) == list:
                xmin=ann['bbox'][0]
                ymin=ann['bbox'][1]
                width=ann['bbox'][2]
                height=ann['bbox'][3]
                x=int(xmin+(width/2))/img['width']
                y=int(ymin+(height/2))/img['height']
                r_width=width/img['width']
                r_height=height/img['height']
                #upper=(int((x*img['width'])-((r_width*img['width'])/2)),int((y*img['height'])-((r_height*img['height'])/2)))
                #lower=(int((x*img['width'])+((r_width*img['width'])/2)),int((y*img['height'])+((r_height*img['height'])/2)))                
                #cv2.rectangle(image, upper, lower ,(128,128,128), thickness = 4)
                f.write(str(ann['category_id']) + ' ' +str(round(x,3)) + ' ' +str(round(y,3)) +' ' + str(round(r_width,3)) + ' ' +str(round(r_height,3)))
                f.write('\n')


# In[ ]:


with open('train.txt','w') as f:
    for i in imgIds_lst:
        path='train2017/'+'0'*(12-len(str(i)))+str(i)+'.jpg'
        f.write(path)
        f.write('\n')

