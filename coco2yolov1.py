
# coding: utf-8

# In[ ]:


import os
import numpy as np
from pycocotools.coco import COCO


# In[ ]:


train_annFile='./annotations/instances_train2017.json'
val_annFile='./annotations/instances_val2017.json'


# In[ ]:


train_coco=COCO(train_annFile)


# In[ ]:


categories=['person','bicycle','car','motorcycle','truck','bus']
catIds_lst =train_coco.getCatIds(catNms=categories)


# In[ ]:


imgIds_lst=list()
for i in catIds_lst:    
    imgIds_lst += train_coco.getImgIds(catIds=i)
imgIds_lst=list(set(imgIds_lst))


# In[ ]:


annIds = train_coco.getAnnIds(imgIds=[262146], catIds=catIds_lst, iscrowd=None)


# In[ ]:


anns = train_coco.loadAnns(annIds)


# In[ ]:


print(imgIds_lst[1:2])


# In[ ]:


img=train_coco.loadImgs(imgIds_lst[1])[0]


# In[ ]:


print(img)


# In[ ]:


anns[0]


# In[ ]:


import cv2
for i in imgIds_lst[0:1]:
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
                f.write(str(ann['category_id']) + ' ' +str(round(x,3)) + ' ' +str(round(y,3)) +' ' + str(round(r_width,3)) + ' ' +str(round(r_height,3)))
                f.write('\n')


# In[64]:


a=3.85764
print(round(a,2))

