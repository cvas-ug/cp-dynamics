Source Code for Paper 'An Ablation Study for Classifying Shapes and Weights of Garments for Robotic Continuous Perception'
-----------------------------------------------
## Description
Step 1. Train a Feature Extractor Model\
\
In our paper, the first step is to train a feature extractor model without an LSTM, trainings of which can be found in Feature_Extractor_Shape.py (for shapes) and Feature_Extractor_Weight.py (for weights). A default learning rate has been set into 0.001 and a defalut epoch has been set into 36. After running the codes, a feature model called alexnet_shape_dict.pth or alexnet_weight_dict.pth can be obtained from the corresponding Feature_Extractor_Shape.py amd Feature_Extractor_Weight.py

Step 2. Train an LSTM Model\
\
After a feature model has been obtained from Step 1, an LSTM model will be trained to learn dynamic changes of garments, trainings of which can be found in LSTM_Shape.py (for shapes) and LSTM_Weight.py (for weights). A default learning rate has been set to 0.0001 and a step scheduler has been utlised. An LSTM model called lstm_shape_dict.pth and lstm_weight_dict.pth will be obatained after running the codes. (corresponding to LSTM_Shape.py and LSTM_Weight.py)

Step3. Evaluate our network by a Moving Average method\
\
When a feature model and an LSTM model are both obtained from trainings in Step 1 and Step 2, an evluation of the network will be made by a method called 'Moving Average'. MA_Shape.py and MA_Weight.py give two confusion matrics over a video sequence of unseen garments on shapes and weights. MS_Shape.py and MS_Weight.py give changes of Moving Averages of different shapes and weights over frames by two trace plotting figures.

---------------------------------------------------
## Dataset
Our database is available at https://bit.ly/3mvuPLJ (for depth images) and https://bit.ly/31X527o (for RGB images). Please place them into 'Database/Real/' file before you train the network.

