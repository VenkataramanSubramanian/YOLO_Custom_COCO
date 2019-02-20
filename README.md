# YOLO_Custom_COCO
Train YOLO on custom coco objects


Step 1: Download COCO dataset and Place the train2017 and val2017 folder in the home

Step 2: Download the COCO annoatations and place it in the home folder

Step 3: Change the filters size in the three layers of the train.cfg to (no.of.classes+5)*3

Step 4: Change the names of the classes you want to train that are present in coco dataset

Step 5: In the  coco2yolo.ipynb line 4 change the categories to the names of the classes you want to train 

Step 6: Update the if-else condition to change the number to from (0,n-1)

Step 7: execute this code in a cmd prompt ./darkent detector train cfg/train.data cfg/train.cfg darknet53.conv.74

(Download the darknet53.conv.74  weigths from wget https://pjreddie.com/media/files/darknet53.conv.74)
