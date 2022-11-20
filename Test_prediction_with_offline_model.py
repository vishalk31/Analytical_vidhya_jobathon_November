#!/usr/bin/env python
# coding: utf-8

# In[4]:


from tensorflow import keras
import joblib

model = keras.models.load_model(r'C:\Users\visha\OneDrive\Documents\Dataset\current_time_series\model_files\final_model.h5')
scaler = joblib.load(r'C:\Users\visha\OneDrive\Documents\Dataset\current_time_series\model_files\input_scaler.pkl')
output_scaler = joblib.load(r'C:\Users\visha\OneDrive\Documents\Dataset\current_time_series\model_files\output_scaler.pkl')
#test_scaled_set = scaler.transform(test_set)


# In[7]:


import pandas as pd
df_test = pd.read_csv(r"C:\Users\visha\OneDrive\Documents\Dataset\current_time_series\Test.csv")
df_test_new = df_test.drop('row_id',axis=1)
df_test_new['datetime'] = pd.to_datetime(df_test_new['datetime'])
df_test_new['day'] = pd.to_datetime(df_test_new['datetime']).dt.day
df_test_new['month'] = pd.to_datetime(df_test_new['datetime']).dt.month
df_test_new['year'] = pd.to_datetime(df_test_new['datetime']).dt.year
df_test_new['Time'] = pd.to_datetime(df_test_new['datetime']).dt.time
df_hrs = df_test_new['Time'].astype(str).str.split(':', expand=True).astype(float)
df_test_new['hours'] = df_hrs[0]
df_test_new.drop(["datetime",'Time'],axis=1,inplace=True)
df_test_min = scaler.transform(df_test_new)
output_result = model.predict(df_test_min)
df_test["energy"] = output_scaler.inverse_transform(output_result)
df_test.drop('datetime',axis=1,inplace=True)
df_test.to_csv(r"C:\Users\visha\OneDrive\Documents\Dataset\current_time_series\submission_22.csv",index=False)


# In[ ]:




