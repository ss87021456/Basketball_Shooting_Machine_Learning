# Basketball_Shooting_Machine_Learning
Using 6-Axis data from hand-back to predict your shooting result
### Dependencies:
python 2.7<br>
numpy <br>
sklearn 
### Training dataset: Features.csv Labels.csv
### Algorithm
Collect 6-Axis data (Gyo-xyz,Acc-xyz) --> Pre-processing --> PCA --> SVM --> Prediction!
![image](https://github.com/ss87021456/Basketball_Shooting_Machine_Learning/blob/master/Algorithm.png)
Cross-Validation get nearly 85% accuarcy
### PCA visualization of training data
![image](https://github.com/ss87021456/Basketball_Shooting_Machine_Learning/blob/master/PCA.png)
### Usage
python train.py
![image](https://github.com/ss87021456/Basketball_Shooting_Machine_Learning/blob/master/train.png)
