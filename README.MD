# Use Library
pip install torch  
pip install xgboost  
pip install sklearn  
pip install maplotlib  
pip install seaborn  

# Code Explanation
1. custom_model.ipynb  
The results of an experment with the model only using numpy

2. XGBoost.ipynb  
The results of XGBoost Regressor and Preprocessing

3. pytorch.ipynb  
The results of Pytorch Sequential model with Linear layers

|Model|Preprocessing|MSE|
|---|:---:|:---:|
|Custom Model|Standard Scaling|0.4620|
|XGBoost|Standard Scaling|**0.3464**|
|Pytorch Linear |Standard Scaling|0.4030|  

4. BestResult.py  
This is the code that summarizes the best way
  
    
# Final Result
XGBoost Regressor + Standard Scaling
|Metric|Score|
|---|---|
|MSE|0.3419|
|RMSE|0.5847|
|MAE|0.0736|
|MAPE|7.3602|
