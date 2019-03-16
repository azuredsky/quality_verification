# 眼底彩照粗筛  
### 1) Requirements  
virtualenv env  
source env/bin/activate  
pip install -r requirements.txt  
### 2）主要依赖：  
python 2.x  
OpenCV 3.1.0 (import cv2能成功就行)  
numpy  1.12.1  
torch  1.0.0  
sklearn 0.20.0  
skimage 0.14.1  

### 3）用法：  
先修改 test_config.json  
运行   python predictor.py  
输出   name_LR_score_final.txt  

### 4）文件夹代码含义：  
../code/  
LR_predict.py:左右眼预测  
data_loader.py：数据加载  
augmentation.py:数据增强处理  
predictor.py:预测眼底彩照质量好坏(默认score>0.5属于符合要求的，score越高，质量越好)  
test_config.json：配置文件，使用时需修改image_root、test_lst_path  
utils.py：可以忽略  

../data_lst/  
图片path保存位置  

../models/  
LR_model.pkl：左右眼模型文件  
quality_model.pkl：预测眼底彩照质量模型文件  

../prediction/  
Prediction_LR_log.txt：左右眼log信息，判断一张图片平均0.2s  
name_LR_score_final.txt：最终生成的结果，依次是图片名称，左右眼，质量得分  

../test_images/  
测试图片  
