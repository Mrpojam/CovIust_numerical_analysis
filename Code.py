# -*- coding: utf-8 -*-
## Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

import pycaret.regression as caret

"""##  Load and Prepare Data"""
data = pd.read_csv("/content/country_vaccinations.csv")
data.head()

"""## Data Preprocessing"""
data.isnull().sum() / data.shape[0]

clean_vaccine_data = data.dropna()
clean_vaccine_data.isnull().sum()

"""## Strorytelling - Visualization"""

clean_vaccine_data['total_vacc'] = np.log10(clean_vaccine_data['total_vaccinations'])
clean_vaccine_data['people_vacc'] = np.log10(clean_vaccine_data['people_vaccinated'])
clean_vaccine_data['people_fully_vacc'] = np.log10(clean_vaccine_data['people_fully_vaccinated'])
clean_vaccine_data['daily_vacc'] = np.log10(clean_vaccine_data['daily_vaccinations'])

clean_vaccine_data = clean_vaccine_data.drop(columns = ['total_vaccinations','people_vaccinated','people_fully_vaccinated', 'daily_vaccinations'])


covid_features = clean_vaccine_data[['date', 'total_vacc', 'people_vacc' , 'people_fully_vacc' , 'daily_vacc']]
sns.set_theme(style="ticks")
sns.pairplot(covid_features)

sub_data = clean_vaccine_data[["country","date","total_vaccinations_per_hundred"]]

print(sub_data.head(5))

US_data = sub_data.loc[sub_data["country"] == "United States"]
Norway_data = sub_data.loc[sub_data["country"] == "Norway"]
Canada_data = sub_data.loc[sub_data["country"] == "Canada"]
UK_data = sub_data.loc[sub_data["country"] == "United Kingdom"]
China_data = sub_data.loc[sub_data["country"] == "China"]
Germany_data = sub_data.loc[sub_data["country"] == "Germany"]
Iran_data = sub_data.loc[sub_data["country"] == "Iran"]

world_countries = pd.concat([US_data, Norway_data, Canada_data, UK_data, China_data, Germany_data, Iran_data], axis = 0)

plt.figure(figsize = (12,6))
plt.title("Total Vaccines per country")
sns.scatterplot(x = world_countries['date'], y = world_countries['total_vaccinations_per_hundred'], hue = world_countries['country'])
ax = plt.gca()
ax.get_yaxis().get_major_formatter().set_scientific(False)
plt.xticks(np.arange(0, len(x), step = 5), rotation = 45)
plt.xlabel("Date")
plt.ylabel("Total Vaccinations")

"""## Train the model"""

data = pd.DataFrame()
data['Date'] = pd.to_datetime(US_data['date'])
data['Target'] = US_data['total_vaccinations_per_hundred']
data.reset_index(drop = True , inplace = True)

data['Series'] = np.arange(1 , len(data)+1)

data['Shift1'] = data.Target.shift(1)

window_len = 10
window = data['Shift1'].rolling(window = window_len)
means = window.mean()
data['Window_mean'] = means


data.dropna(inplace = True)
data.reset_index(drop = True , inplace=True)

dates = data['Date'] 

data = data[['Series' , 'Window_mean' , 'Shift1' , 'Target']]

data.isnull().sum()

data.shape

sns.scatterplot(data=data, x='Series', y='Target')

train = data.iloc[:292,:] 
test = data.iloc[292:,:]

setup = caret.setup(data = data, test_data = test ,target = 'Target' , fold_strategy = 'timeseries'
                 , remove_perfect_collinearity = False , numeric_features = ['Series' , 'Window_mean' , 'Shift1'] 
                     , fold = 5 , session_id = 51)

best = caret.compare_models(sort = 'MAE' , turbo = False)

best = caret.create_model('ridge')

_ = caret.predict_model(best)

"""## Test Results"""

future = pd.DataFrame(columns = ['Series' , 'Window_mean' , 'Shift1'])
future['Series'] = np.arange(191,341) 
future['Window_mean'] = np.nan
future['Shift1'] = np.nan

future.iloc[0,2] = data['Target'].max()
sum = 0
for i in range(window_len):
    sum += data.iloc[len(data)-1-i,3]
    
future.iloc[0,1] = sum/window_len
future.shape

for j in range(len(future)):
    current_row = j
    next_row = j+1
    
    
    if current_row != len(future)-1 :
        #print(current_row, next_row)
        pr = caret.predict_model(best , future.iloc[[current_row]])['Label']
        future.iloc[next_row,2] = float(pr)
        
       # print(future.iloc[next_row,2]-future.iloc[current_row,2])
        
        
        if next_row < 9 :
            sum = 0
            num_rows_from_data = window_len - (next_row + 1)
            num_rows_from_future = window_len - num_rows_from_data

            for i in range(num_rows_from_data):
                sum += data.iloc[len(data)-1-i , 2]


            for i in range(num_rows_from_future):
                sum += future.iloc[next_row - i , 2]

            future.iloc[next_row , 1] = sum/window_len


        elif next_row >= 9:
            sum = 0
            for i in range(window_len):
                sum += future.iloc[next_row-i,2]
            future.iloc[next_row,1] = sum/window_len

from datetime import date , datetime , timedelta

future['Predicted'] = future['Shift1'].shift(-1)

start = datetime.strptime("2021-12-26", "%Y-%m-%d")
date_generated = [start + timedelta(days=x) for x in range(0, 150)]
date_list = []
for date in date_generated:
    date_list.append(date.strftime("%Y-%m-%d"))
    
future['Date'] = date_list

future = future[['Date' , 'Predicted']]
future.dropna(inplace = True)
future['Predicted'] += 25

import plotly.graph_objects as go
fig = go.Figure(data=go.Scatter(x=Canada_data['date'], y = Canada_data['total_vaccinations_per_hundred']
                                ,mode='lines', line_color='red' , name = 'Until now'))
fig.add_trace(go.Scatter(x=future['Date'], y=future['Predicted'],mode='lines', line=dict(color="#0000ff"), name = 'Future'))

fig.update_layout(template = 'plotly_dark')

fig.show()

caret.save_model(best, 'CanadaVaccination')