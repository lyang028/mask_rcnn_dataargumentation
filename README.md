# Mask R-CNN for Object Detection and Segmentation

This is a modified version of Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow (https://github.com/matterport/Mask_RCNN.git). It is modified to detect the worker pose in construction site. 

We provide 3 levels, 5 kinds of data in this repository.

Level-0 data is the images collectedfrom the construction site directly without changing whose average entropy is(5.76,11.85).
Level-1 data is the silhouette generated manually based onthe real images whose average entropy is(0.56,1.28).
Level-2 data is the posture simulatedby the stick-man based on our common sense whose aver-age entropy is(0.48,0.96).
2 sub-levels of data with the feature. Their average entropy are(0.95,2.12)and(0.76,1.38).

![image](https://github.com/lyang028/mask_rcnn_dataargumentation/figure/level.bmp)

Also, we provide two test dataset.
![image](https://github.com/lyang028/mask_rcnn_dataargumentation/figure/test.bmp)

You can use worker_keep_train.py to train the model with one of the dataset with the guidence of the console output and use prediction.py to test the performance.

A related Colab notebook is provided. You can mount your google drive and access it directly.
