# Analytical_vidhya_jobathon_November
Exploratory Data Analysis (EDA) and Data Pre Processing
●	First Step is to Find Missing Values in data (There are 1200 data missing values in energy columns in the dataset)

●	Replacing the missing values in the column by pandas ffill (Forward Fill)( https://www.geeksforgeeks.org/python-pandas-dataframe-ffill/)

●	Single column Date has been divided into Day, month, year, hour separate columns 
■	Example (31-07-2003 13:00:00)   31 07 2003 13

●	Once Segmentation of Date is done next step is feature scaling. Minmax scaling is taken into consideration in this model. Purpose of Scaling is to make minimize the spread of data
(https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)

●	Scaling have been applied to both input and output variable for LSTM Model
●	FB Prophet does not require any changes in the data set it could be applied directly into the model

Train ML Algorithm and Optimization.
●	LSTM and FB prophet model were used for this problem
●	Final output is the mean of both predicted results
●	Multiple variants of LSTM and RNN model were taken into consideration and finally a single layer LSTM with 200 units batch size of (16) and epochs 20 gave me a better result  
●	Prophet by default performed well the data set no hyper parameter required. 
Test The ML Algorithm
●	Prepare your Test data the way you prepared test data
●	Predict using the trained model 
●	Convert into csv and submit 
Entire Code:
    
●	Entire code with models have been uploaded into the GitHub pages also multiple LSTM models and RNN models are also available in the GitHub page below

