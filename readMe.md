This project aims to predict the HKAA from full-limb X-ray.
In this project we leveraged YOLOv3 and U-Net model for roi extraction and landmark prediction.
By using YOLOv3 model, we extracted the ROI of hip, knee and ankle joint.
Afte that, we applied U-Net model for heatmap regression on each of the three joint to predict the center of the joint.
Then, we transformed the predicted heatmap into coordinate's information and mapped it back to original X-ray.
Finally, we calculated HKAA from predicted coordiantes and compared with our annotation.
The results shows that our method can archieve a high accuracy of HKAA prediction.
