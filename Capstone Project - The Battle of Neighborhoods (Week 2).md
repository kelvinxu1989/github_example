### Part 1:Using a real world data set from Kaggle containing the Vancouver Crimes from 2003 to 2019<a name="part1"></a>


####  Vancouver Crime Report 

Properties of the Crime Report

*   TYPE - Crime type
*   YEAR - Recorded year
*   MONTH - Recorded month
*   DAY - Recorded day
*   HOUR - Recorded hour
*   MINUTE - Recorded minute
*   HUNDRED_BLOCK - Recorded block
*   NEIGHBOURHOOD - Recorded neighborhood
*   X - GPS longtitude
*   Y - GPS latitude

Data set URL: https://www.kaggle.com/agilesifaka/vancouver-crime-report/version/2

### Importing all the necessary Libraries


```python
import numpy as np
import pandas as pd

#Command to install OpenCage Geocoder for fetching Lat and Lng of Neighborhood
!pip install opencage

#Importing OpenCage Geocoder
from opencage.geocoder import OpenCageGeocode

# use the inline backend to generate the plots within the browser
%matplotlib inline 

#Importing Matplot lib and associated packages to perform Data Visualisation and Exploratory Data Analysis
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use('ggplot') # optional: for ggplot-like style

# check for latest version of Matplotlib
print ('Matplotlib version: ', mpl.__version__) # >= 2.0.0

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

#Importing folium to visualise Maps and plot based on Lat and Lng
import folium

#Requests to request web pages by making get requests to FourSquare REST Client
import requests

#To normalise data returned by FourSquare API
from pandas.io.json import json_normalize

#Importing KMeans from SciKit library to Classify neighborhoods into clusters
from sklearn.cluster import KMeans

print('Libraries imported')
```

    Collecting opencage
      Downloading https://files.pythonhosted.org/packages/00/6b/05922eb2ea69713f3c9e355649d8c905a7a0880e9511b7b10d6dedeb859e/opencage-1.2.1-py3-none-any.whl
    Requirement already satisfied: pyopenssl>=0.15.1 in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from opencage) (19.1.0)
    Collecting backoff>=1.10.0 (from opencage)
      Downloading https://files.pythonhosted.org/packages/f0/32/c5dd4f4b0746e9ec05ace2a5045c1fc375ae67ee94355344ad6c7005fd87/backoff-1.10.0-py2.py3-none-any.whl
    Requirement already satisfied: Requests>=2.2.0 in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from opencage) (2.23.0)
    Requirement already satisfied: six>=1.4.0 in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from opencage) (1.14.0)
    Requirement already satisfied: cryptography>=2.8 in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from pyopenssl>=0.15.1->opencage) (2.9.2)
    Requirement already satisfied: certifi>=2017.4.17 in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from Requests>=2.2.0->opencage) (2020.4.5.1)
    Requirement already satisfied: chardet<4,>=3.0.2 in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from Requests>=2.2.0->opencage) (3.0.4)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from Requests>=2.2.0->opencage) (1.25.9)
    Requirement already satisfied: idna<3,>=2.5 in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from Requests>=2.2.0->opencage) (2.9)
    Requirement already satisfied: cffi!=1.11.3,>=1.8 in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from cryptography>=2.8->pyopenssl>=0.15.1->opencage) (1.14.0)
    Requirement already satisfied: pycparser in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from cffi!=1.11.3,>=1.8->cryptography>=2.8->pyopenssl>=0.15.1->opencage) (2.20)
    Installing collected packages: backoff, opencage
    Successfully installed backoff-1.10.0 opencage-1.2.1
    Matplotlib version:  3.1.1
    Libraries imported


### Reading from the Dataset

###### Due to sheer amount of data(~ 600,000 rows), it was not possible to process all of them and instead for this project we will be considering the recent crime report of the 2018.


```python
vnc_crime_df = pd.read_csv('https://raw.githubusercontent.com/RamanujaSVL/Coursera_Capstone/master/vancouver_crime_records_2018.csv', index_col=None)

#Dropping X,Y which represents Lat, Lng data as Coordinates, the data seems to be corrupt
vnc_crime_df.drop(['Unnamed: 0','MINUTE', 'HUNDRED_BLOCK', 'X', 'Y'], axis = 1, inplace = True)

#vnc_crime_df.columns

vnc_crime_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TYPE</th>
      <th>YEAR</th>
      <th>MONTH</th>
      <th>DAY</th>
      <th>HOUR</th>
      <th>NEIGHBOURHOOD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Break and Enter Commercial</td>
      <td>2018</td>
      <td>3</td>
      <td>2</td>
      <td>6</td>
      <td>West End</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Break and Enter Commercial</td>
      <td>2018</td>
      <td>6</td>
      <td>16</td>
      <td>18</td>
      <td>West End</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Break and Enter Commercial</td>
      <td>2018</td>
      <td>12</td>
      <td>12</td>
      <td>0</td>
      <td>West End</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Break and Enter Commercial</td>
      <td>2018</td>
      <td>4</td>
      <td>9</td>
      <td>6</td>
      <td>Central Business District</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Break and Enter Commercial</td>
      <td>2018</td>
      <td>10</td>
      <td>2</td>
      <td>18</td>
      <td>Central Business District</td>
    </tr>
  </tbody>
</table>
</div>



#### Changing the name of columns to lowercase


```python
vnc_crime_df.columns = ['Type', 'Year','Month','Day','Hour','Neighbourhood']
vnc_crime_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Type</th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
      <th>Hour</th>
      <th>Neighbourhood</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Break and Enter Commercial</td>
      <td>2018</td>
      <td>3</td>
      <td>2</td>
      <td>6</td>
      <td>West End</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Break and Enter Commercial</td>
      <td>2018</td>
      <td>6</td>
      <td>16</td>
      <td>18</td>
      <td>West End</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Break and Enter Commercial</td>
      <td>2018</td>
      <td>12</td>
      <td>12</td>
      <td>0</td>
      <td>West End</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Break and Enter Commercial</td>
      <td>2018</td>
      <td>4</td>
      <td>9</td>
      <td>6</td>
      <td>Central Business District</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Break and Enter Commercial</td>
      <td>2018</td>
      <td>10</td>
      <td>2</td>
      <td>18</td>
      <td>Central Business District</td>
    </tr>
  </tbody>
</table>
</div>



### Total Crimes in different Neighborhoods


```python
vnc_crime_df['Neighbourhood'].value_counts()
```




    Central Business District    10857
    West End                      3031
    Mount Pleasant                2396
    Strathcona                    1987
    Kitsilano                     1802
    Fairview                      1795
    Renfrew-Collingwood           1762
    Grandview-Woodland            1761
    Kensington-Cedar Cottage      1391
    Hastings-Sunrise              1270
    Sunset                         967
    Riley Park                     866
    Marpole                        828
    Victoria-Fraserview            600
    Killarney                      565
    Oakridge                       499
    Dunbar-Southlands              474
    Kerrisdale                     417
    Shaughnessy                    414
    West Point Grey                372
    Arbutus Ridge                  311
    South Cambie                   292
    Stanley Park                   154
    Musqueam                        17
    Name: Neighbourhood, dtype: int64



### Part 2:Gathering additional information about the Neighborhood from Wikipedia<a name="part2"></a>

#### As part of data set Borough which the neighborhood was part of was not categorized, so we will create a dictionary of Neighborhood and based on data in the following [Wikipedia page](https://en.wikipedia.org/wiki/List_of_neighbourhoods_in_Vancouver).


```python
# define the dataframe columns
column_names = ['Neighbourhood', 'Borough'] 

# instantiate the dataframe
vnc_neigh_bor = pd.DataFrame(columns=column_names)

vnc_neigh_bor['Neighbourhood'] = vnc_crime_df['Neighbourhood'].unique()

neigh_bor_dict = {'Central Business District':'Central', 'West End':'Central', 'Stanley Park':'Central', 'Victoria-Fraserview':'South Vancouver',
                  'Killarney':'South Vancouver', 'Musqueam':'South Vancouver', 'Mount Pleasant':'East Side', 'Strathcona':'East Side',
                  'Renfrew-Collingwood':'East Side', 'Grandview-Woodland':'East Side', 'Kensington-Cedar Cottage':'East Side', 'Hastings-Sunrise':'East Side',
                  'Sunset':'East Side', 'Riley Park':'East Side', 'Kitsilano':'West Side', 'Fairview':'West Side',
                  'Marpole':'West Side', 'Oakridge':'West Side', 'Dunbar-Southlands':'West Side', 'Kerrisdale':'West Side',
                  'Shaughnessy':'West Side', 'West Point Grey':'West Side', 'Arbutus Ridge':'West Side', 'South Cambie':'West Side'}

for row, neigh in zip(neigh_bor_dict, vnc_neigh_bor['Neighbourhood']):
  vnc_neigh_bor.loc[vnc_neigh_bor.Neighbourhood == row, 'Borough'] = neigh_bor_dict.get(row)

vnc_neigh_bor.dropna(inplace=True)

print("Total Neighbourhood Count",len(vnc_neigh_bor['Neighbourhood']),"Borough Count",len(vnc_neigh_bor['Borough'].unique()))

vnc_neigh_bor.head()
```

    Total Neighbourhood Count 24 Borough Count 4





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighbourhood</th>
      <th>Borough</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>West End</td>
      <td>Central</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Central Business District</td>
      <td>Central</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Hastings-Sunrise</td>
      <td>East Side</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Grandview-Woodland</td>
      <td>East Side</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Mount Pleasant</td>
      <td>East Side</td>
    </tr>
  </tbody>
</table>
</div>



### Merging the Crime data Table to include Boroughs


```python
vnc_boroughs_crime = pd.merge(vnc_crime_df,vnc_neigh_bor, on='Neighbourhood')

vnc_boroughs_crime.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Type</th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
      <th>Hour</th>
      <th>Neighbourhood</th>
      <th>Borough</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Break and Enter Commercial</td>
      <td>2018</td>
      <td>3</td>
      <td>2</td>
      <td>6</td>
      <td>West End</td>
      <td>Central</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Break and Enter Commercial</td>
      <td>2018</td>
      <td>6</td>
      <td>16</td>
      <td>18</td>
      <td>West End</td>
      <td>Central</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Break and Enter Commercial</td>
      <td>2018</td>
      <td>12</td>
      <td>12</td>
      <td>0</td>
      <td>West End</td>
      <td>Central</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Break and Enter Commercial</td>
      <td>2018</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>West End</td>
      <td>Central</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Break and Enter Commercial</td>
      <td>2018</td>
      <td>3</td>
      <td>17</td>
      <td>11</td>
      <td>West End</td>
      <td>Central</td>
    </tr>
  </tbody>
</table>
</div>



##### Further Cleaning the data by dropping rows with invalid data


```python
vnc_boroughs_crime.dropna(inplace=True)
vnc_boroughs_crime['Borough'].value_counts()
```




    Central            14042
    East Side          12400
    West Side           7204
    South Vancouver     1182
    Name: Borough, dtype: int64



### Methodology<a name="methodology"></a>

Categorized the methodologysection into two parts:

- [**Exploratory Data Analysis**:](#eda) Visualise the crime repots in different Vancouver boroughs to idenity the safest borough and normalise the neighborhoods of that borough. We will Use the resulting data and find 10 most common venues in each neighborhood.


- [**Modelling**:](#mdl) To help stakeholders choose the right neighborhood within a borough we will be clustering similar neighborhoods using K - means clustering which is a form of unsupervised machine learning algorithm that clusters data based on predefined cluster size. We will use K-Means clustering to address this problem so as to group data based on existing venues which will help in the decision making process.

#### Exploratory Data Analysis

#### Pivoting the table to better understand the data by crimes per borough

> Indented block


```python
vnc_crime_cat = pd.pivot_table(vnc_boroughs_crime,
                               values=['Year'],
                               index=['Borough'],
                               columns=['Type'],
                               aggfunc=len,
                               fill_value=0,
                               margins=True)
vnc_crime_cat
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="10" halign="left">Year</th>
    </tr>
    <tr>
      <th>Type</th>
      <th>Break and Enter Commercial</th>
      <th>Break and Enter Residential/Other</th>
      <th>Mischief</th>
      <th>Other Theft</th>
      <th>Theft from Vehicle</th>
      <th>Theft of Bicycle</th>
      <th>Theft of Vehicle</th>
      <th>Vehicle Collision or Pedestrian Struck (with Fatality)</th>
      <th>Vehicle Collision or Pedestrian Struck (with Injury)</th>
      <th>All</th>
    </tr>
    <tr>
      <th>Borough</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Central</th>
      <td>787</td>
      <td>198</td>
      <td>2280</td>
      <td>2489</td>
      <td>6871</td>
      <td>857</td>
      <td>245</td>
      <td>1</td>
      <td>314</td>
      <td>14042</td>
    </tr>
    <tr>
      <th>East Side</th>
      <td>786</td>
      <td>1043</td>
      <td>2192</td>
      <td>1674</td>
      <td>4754</td>
      <td>678</td>
      <td>605</td>
      <td>8</td>
      <td>660</td>
      <td>12400</td>
    </tr>
    <tr>
      <th>South Vancouver</th>
      <td>49</td>
      <td>156</td>
      <td>187</td>
      <td>88</td>
      <td>483</td>
      <td>36</td>
      <td>71</td>
      <td>1</td>
      <td>111</td>
      <td>1182</td>
    </tr>
    <tr>
      <th>West Side</th>
      <td>403</td>
      <td>1000</td>
      <td>1062</td>
      <td>696</td>
      <td>2838</td>
      <td>588</td>
      <td>225</td>
      <td>3</td>
      <td>389</td>
      <td>7204</td>
    </tr>
    <tr>
      <th>All</th>
      <td>2025</td>
      <td>2397</td>
      <td>5721</td>
      <td>4947</td>
      <td>14946</td>
      <td>2159</td>
      <td>1146</td>
      <td>13</td>
      <td>1474</td>
      <td>34828</td>
    </tr>
  </tbody>
</table>
</div>



##### Merging the Pivoted Column with other columns


```python
vnc_crime_cat.reset_index(inplace = True)
vnc_crime_cat.columns = vnc_crime_cat.columns.map(''.join)
vnc_crime_cat.rename(columns={'YearAll':'Total'}, inplace=True)
# To ignore bottom All in Borough
vnc_crime_cat = vnc_crime_cat.head(4)
vnc_crime_cat
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough</th>
      <th>YearBreak and Enter Commercial</th>
      <th>YearBreak and Enter Residential/Other</th>
      <th>YearMischief</th>
      <th>YearOther Theft</th>
      <th>YearTheft from Vehicle</th>
      <th>YearTheft of Bicycle</th>
      <th>YearTheft of Vehicle</th>
      <th>YearVehicle Collision or Pedestrian Struck (with Fatality)</th>
      <th>YearVehicle Collision or Pedestrian Struck (with Injury)</th>
      <th>Total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Central</td>
      <td>787</td>
      <td>198</td>
      <td>2280</td>
      <td>2489</td>
      <td>6871</td>
      <td>857</td>
      <td>245</td>
      <td>1</td>
      <td>314</td>
      <td>14042</td>
    </tr>
    <tr>
      <th>1</th>
      <td>East Side</td>
      <td>786</td>
      <td>1043</td>
      <td>2192</td>
      <td>1674</td>
      <td>4754</td>
      <td>678</td>
      <td>605</td>
      <td>8</td>
      <td>660</td>
      <td>12400</td>
    </tr>
    <tr>
      <th>2</th>
      <td>South Vancouver</td>
      <td>49</td>
      <td>156</td>
      <td>187</td>
      <td>88</td>
      <td>483</td>
      <td>36</td>
      <td>71</td>
      <td>1</td>
      <td>111</td>
      <td>1182</td>
    </tr>
    <tr>
      <th>3</th>
      <td>West Side</td>
      <td>403</td>
      <td>1000</td>
      <td>1062</td>
      <td>696</td>
      <td>2838</td>
      <td>588</td>
      <td>225</td>
      <td>3</td>
      <td>389</td>
      <td>7204</td>
    </tr>
  </tbody>
</table>
</div>



#### Pivoting the table to better understand the data by crimes per neighborhood


```python
vnc_crime_neigh = pd.pivot_table(vnc_boroughs_crime,
                               values=['Year'],
                               index=['Neighbourhood'],
                               columns=['Type'],
                               aggfunc=len,
                               fill_value=0,
                               margins=True)
vnc_crime_neigh
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="10" halign="left">Year</th>
    </tr>
    <tr>
      <th>Type</th>
      <th>Break and Enter Commercial</th>
      <th>Break and Enter Residential/Other</th>
      <th>Mischief</th>
      <th>Other Theft</th>
      <th>Theft from Vehicle</th>
      <th>Theft of Bicycle</th>
      <th>Theft of Vehicle</th>
      <th>Vehicle Collision or Pedestrian Struck (with Fatality)</th>
      <th>Vehicle Collision or Pedestrian Struck (with Injury)</th>
      <th>All</th>
    </tr>
    <tr>
      <th>Neighbourhood</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Arbutus Ridge</th>
      <td>12</td>
      <td>78</td>
      <td>49</td>
      <td>18</td>
      <td>111</td>
      <td>12</td>
      <td>12</td>
      <td>1</td>
      <td>18</td>
      <td>311</td>
    </tr>
    <tr>
      <th>Central Business District</th>
      <td>551</td>
      <td>124</td>
      <td>1812</td>
      <td>2034</td>
      <td>5301</td>
      <td>640</td>
      <td>165</td>
      <td>0</td>
      <td>230</td>
      <td>10857</td>
    </tr>
    <tr>
      <th>Dunbar-Southlands</th>
      <td>8</td>
      <td>106</td>
      <td>81</td>
      <td>31</td>
      <td>199</td>
      <td>16</td>
      <td>9</td>
      <td>1</td>
      <td>23</td>
      <td>474</td>
    </tr>
    <tr>
      <th>Fairview</th>
      <td>138</td>
      <td>73</td>
      <td>233</td>
      <td>297</td>
      <td>692</td>
      <td>245</td>
      <td>55</td>
      <td>0</td>
      <td>62</td>
      <td>1795</td>
    </tr>
    <tr>
      <th>Grandview-Woodland</th>
      <td>148</td>
      <td>162</td>
      <td>304</td>
      <td>215</td>
      <td>634</td>
      <td>110</td>
      <td>123</td>
      <td>0</td>
      <td>65</td>
      <td>1761</td>
    </tr>
    <tr>
      <th>Hastings-Sunrise</th>
      <td>48</td>
      <td>117</td>
      <td>195</td>
      <td>107</td>
      <td>607</td>
      <td>52</td>
      <td>74</td>
      <td>0</td>
      <td>70</td>
      <td>1270</td>
    </tr>
    <tr>
      <th>Kensington-Cedar Cottage</th>
      <td>62</td>
      <td>145</td>
      <td>255</td>
      <td>148</td>
      <td>541</td>
      <td>69</td>
      <td>71</td>
      <td>3</td>
      <td>97</td>
      <td>1391</td>
    </tr>
    <tr>
      <th>Kerrisdale</th>
      <td>24</td>
      <td>97</td>
      <td>49</td>
      <td>9</td>
      <td>172</td>
      <td>13</td>
      <td>11</td>
      <td>0</td>
      <td>42</td>
      <td>417</td>
    </tr>
    <tr>
      <th>Killarney</th>
      <td>34</td>
      <td>72</td>
      <td>90</td>
      <td>31</td>
      <td>240</td>
      <td>19</td>
      <td>33</td>
      <td>0</td>
      <td>46</td>
      <td>565</td>
    </tr>
    <tr>
      <th>Kitsilano</th>
      <td>106</td>
      <td>165</td>
      <td>320</td>
      <td>154</td>
      <td>755</td>
      <td>189</td>
      <td>51</td>
      <td>1</td>
      <td>61</td>
      <td>1802</td>
    </tr>
    <tr>
      <th>Marpole</th>
      <td>44</td>
      <td>125</td>
      <td>134</td>
      <td>75</td>
      <td>290</td>
      <td>34</td>
      <td>39</td>
      <td>0</td>
      <td>87</td>
      <td>828</td>
    </tr>
    <tr>
      <th>Mount Pleasant</th>
      <td>205</td>
      <td>124</td>
      <td>353</td>
      <td>493</td>
      <td>822</td>
      <td>232</td>
      <td>67</td>
      <td>0</td>
      <td>100</td>
      <td>2396</td>
    </tr>
    <tr>
      <th>Musqueam</th>
      <td>0</td>
      <td>4</td>
      <td>3</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>17</td>
    </tr>
    <tr>
      <th>Oakridge</th>
      <td>19</td>
      <td>123</td>
      <td>64</td>
      <td>63</td>
      <td>164</td>
      <td>18</td>
      <td>18</td>
      <td>0</td>
      <td>30</td>
      <td>499</td>
    </tr>
    <tr>
      <th>Renfrew-Collingwood</th>
      <td>91</td>
      <td>156</td>
      <td>243</td>
      <td>472</td>
      <td>569</td>
      <td>37</td>
      <td>92</td>
      <td>0</td>
      <td>102</td>
      <td>1762</td>
    </tr>
    <tr>
      <th>Riley Park</th>
      <td>35</td>
      <td>122</td>
      <td>140</td>
      <td>53</td>
      <td>378</td>
      <td>52</td>
      <td>39</td>
      <td>2</td>
      <td>45</td>
      <td>866</td>
    </tr>
    <tr>
      <th>Shaughnessy</th>
      <td>12</td>
      <td>120</td>
      <td>41</td>
      <td>0</td>
      <td>187</td>
      <td>10</td>
      <td>11</td>
      <td>0</td>
      <td>33</td>
      <td>414</td>
    </tr>
    <tr>
      <th>South Cambie</th>
      <td>22</td>
      <td>42</td>
      <td>41</td>
      <td>38</td>
      <td>111</td>
      <td>19</td>
      <td>8</td>
      <td>0</td>
      <td>11</td>
      <td>292</td>
    </tr>
    <tr>
      <th>Stanley Park</th>
      <td>6</td>
      <td>2</td>
      <td>8</td>
      <td>0</td>
      <td>109</td>
      <td>14</td>
      <td>3</td>
      <td>0</td>
      <td>12</td>
      <td>154</td>
    </tr>
    <tr>
      <th>Strathcona</th>
      <td>160</td>
      <td>124</td>
      <td>527</td>
      <td>81</td>
      <td>821</td>
      <td>108</td>
      <td>76</td>
      <td>2</td>
      <td>88</td>
      <td>1987</td>
    </tr>
    <tr>
      <th>Sunset</th>
      <td>37</td>
      <td>93</td>
      <td>175</td>
      <td>105</td>
      <td>382</td>
      <td>18</td>
      <td>63</td>
      <td>1</td>
      <td>93</td>
      <td>967</td>
    </tr>
    <tr>
      <th>Victoria-Fraserview</th>
      <td>15</td>
      <td>80</td>
      <td>94</td>
      <td>57</td>
      <td>239</td>
      <td>15</td>
      <td>36</td>
      <td>1</td>
      <td>63</td>
      <td>600</td>
    </tr>
    <tr>
      <th>West End</th>
      <td>230</td>
      <td>72</td>
      <td>460</td>
      <td>455</td>
      <td>1461</td>
      <td>203</td>
      <td>77</td>
      <td>1</td>
      <td>72</td>
      <td>3031</td>
    </tr>
    <tr>
      <th>West Point Grey</th>
      <td>18</td>
      <td>71</td>
      <td>50</td>
      <td>11</td>
      <td>157</td>
      <td>32</td>
      <td>11</td>
      <td>0</td>
      <td>22</td>
      <td>372</td>
    </tr>
    <tr>
      <th>All</th>
      <td>2025</td>
      <td>2397</td>
      <td>5721</td>
      <td>4947</td>
      <td>14946</td>
      <td>2159</td>
      <td>1146</td>
      <td>13</td>
      <td>1474</td>
      <td>34828</td>
    </tr>
  </tbody>
</table>
</div>



##### Merging the Pivoted Column with other columns

*   List item
*   List item


```python
vnc_crime_neigh.reset_index(inplace = True)
vnc_crime_neigh.columns = vnc_crime_neigh.columns.map(''.join)
vnc_crime_neigh.rename(columns={'YearAll':'Total'}, inplace=True)

vnc_crime_neigh.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighbourhood</th>
      <th>YearBreak and Enter Commercial</th>
      <th>YearBreak and Enter Residential/Other</th>
      <th>YearMischief</th>
      <th>YearOther Theft</th>
      <th>YearTheft from Vehicle</th>
      <th>YearTheft of Bicycle</th>
      <th>YearTheft of Vehicle</th>
      <th>YearVehicle Collision or Pedestrian Struck (with Fatality)</th>
      <th>YearVehicle Collision or Pedestrian Struck (with Injury)</th>
      <th>Total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Arbutus Ridge</td>
      <td>12</td>
      <td>78</td>
      <td>49</td>
      <td>18</td>
      <td>111</td>
      <td>12</td>
      <td>12</td>
      <td>1</td>
      <td>18</td>
      <td>311</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Central Business District</td>
      <td>551</td>
      <td>124</td>
      <td>1812</td>
      <td>2034</td>
      <td>5301</td>
      <td>640</td>
      <td>165</td>
      <td>0</td>
      <td>230</td>
      <td>10857</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Dunbar-Southlands</td>
      <td>8</td>
      <td>106</td>
      <td>81</td>
      <td>31</td>
      <td>199</td>
      <td>16</td>
      <td>9</td>
      <td>1</td>
      <td>23</td>
      <td>474</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Fairview</td>
      <td>138</td>
      <td>73</td>
      <td>233</td>
      <td>297</td>
      <td>692</td>
      <td>245</td>
      <td>55</td>
      <td>0</td>
      <td>62</td>
      <td>1795</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Grandview-Woodland</td>
      <td>148</td>
      <td>162</td>
      <td>304</td>
      <td>215</td>
      <td>634</td>
      <td>110</td>
      <td>123</td>
      <td>0</td>
      <td>65</td>
      <td>1761</td>
    </tr>
  </tbody>
</table>
</div>



##### Pandas describe() is used to view some basic statistical details like percentile, mean, std etc. of a data frame or a series of numeric values.


```python
vnc_crime_cat.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>YearBreak and Enter Commercial</th>
      <th>YearBreak and Enter Residential/Other</th>
      <th>YearMischief</th>
      <th>YearOther Theft</th>
      <th>YearTheft from Vehicle</th>
      <th>YearTheft of Bicycle</th>
      <th>YearTheft of Vehicle</th>
      <th>YearVehicle Collision or Pedestrian Struck (with Fatality)</th>
      <th>YearVehicle Collision or Pedestrian Struck (with Injury)</th>
      <th>Total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.00000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>506.250000</td>
      <td>599.250000</td>
      <td>1430.25000</td>
      <td>1236.750000</td>
      <td>3736.500000</td>
      <td>539.750000</td>
      <td>286.500000</td>
      <td>3.250000</td>
      <td>368.500000</td>
      <td>8707.000000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>354.409721</td>
      <td>488.189427</td>
      <td>997.26572</td>
      <td>1060.087221</td>
      <td>2723.536977</td>
      <td>353.955153</td>
      <td>226.117226</td>
      <td>3.304038</td>
      <td>227.060198</td>
      <td>5801.870618</td>
    </tr>
    <tr>
      <th>min</th>
      <td>49.000000</td>
      <td>156.000000</td>
      <td>187.00000</td>
      <td>88.000000</td>
      <td>483.000000</td>
      <td>36.000000</td>
      <td>71.000000</td>
      <td>1.000000</td>
      <td>111.000000</td>
      <td>1182.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>314.500000</td>
      <td>187.500000</td>
      <td>843.25000</td>
      <td>544.000000</td>
      <td>2249.250000</td>
      <td>450.000000</td>
      <td>186.500000</td>
      <td>1.000000</td>
      <td>263.250000</td>
      <td>5698.500000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>594.500000</td>
      <td>599.000000</td>
      <td>1627.00000</td>
      <td>1185.000000</td>
      <td>3796.000000</td>
      <td>633.000000</td>
      <td>235.000000</td>
      <td>2.000000</td>
      <td>351.500000</td>
      <td>9802.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>786.250000</td>
      <td>1010.750000</td>
      <td>2214.00000</td>
      <td>1877.750000</td>
      <td>5283.250000</td>
      <td>722.750000</td>
      <td>335.000000</td>
      <td>4.250000</td>
      <td>456.750000</td>
      <td>12810.500000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>787.000000</td>
      <td>1043.000000</td>
      <td>2280.00000</td>
      <td>2489.000000</td>
      <td>6871.000000</td>
      <td>857.000000</td>
      <td>605.000000</td>
      <td>8.000000</td>
      <td>660.000000</td>
      <td>14042.000000</td>
    </tr>
  </tbody>
</table>
</div>



#### Expolring the data by Visualising

##### Sorting the data by crimes per neighborhood

> Indented block


```python
vnc_crime_neigh.sort_values(['Total'], ascending = False, axis = 0, inplace = True )

crime_neigh_top5 = vnc_crime_neigh.iloc[1:6]
crime_neigh_top5
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighbourhood</th>
      <th>YearBreak and Enter Commercial</th>
      <th>YearBreak and Enter Residential/Other</th>
      <th>YearMischief</th>
      <th>YearOther Theft</th>
      <th>YearTheft from Vehicle</th>
      <th>YearTheft of Bicycle</th>
      <th>YearTheft of Vehicle</th>
      <th>YearVehicle Collision or Pedestrian Struck (with Fatality)</th>
      <th>YearVehicle Collision or Pedestrian Struck (with Injury)</th>
      <th>Total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Central Business District</td>
      <td>551</td>
      <td>124</td>
      <td>1812</td>
      <td>2034</td>
      <td>5301</td>
      <td>640</td>
      <td>165</td>
      <td>0</td>
      <td>230</td>
      <td>10857</td>
    </tr>
    <tr>
      <th>22</th>
      <td>West End</td>
      <td>230</td>
      <td>72</td>
      <td>460</td>
      <td>455</td>
      <td>1461</td>
      <td>203</td>
      <td>77</td>
      <td>1</td>
      <td>72</td>
      <td>3031</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Mount Pleasant</td>
      <td>205</td>
      <td>124</td>
      <td>353</td>
      <td>493</td>
      <td>822</td>
      <td>232</td>
      <td>67</td>
      <td>0</td>
      <td>100</td>
      <td>2396</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Strathcona</td>
      <td>160</td>
      <td>124</td>
      <td>527</td>
      <td>81</td>
      <td>821</td>
      <td>108</td>
      <td>76</td>
      <td>2</td>
      <td>88</td>
      <td>1987</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Kitsilano</td>
      <td>106</td>
      <td>165</td>
      <td>320</td>
      <td>154</td>
      <td>755</td>
      <td>189</td>
      <td>51</td>
      <td>1</td>
      <td>61</td>
      <td>1802</td>
    </tr>
  </tbody>
</table>
</div>



##### Five Neighborhoods with highest crime

> Indented block


```python
per_neigh = crime_neigh_top5[['Neighbourhood','Total']]

per_neigh.set_index('Neighbourhood',inplace = True)

ax = per_neigh.plot(kind='bar', figsize=(10, 6), rot=0)

ax.set_ylabel('Number of Crimes')
ax.set_xlabel('Neighbourhood')
ax.set_title('Neighbourhoods in Vancouver with the Highest crimes')

for p in ax.patches:
    ax.annotate(np.round(p.get_height(),decimals=2), 
                (p.get_x()+p.get_width()/2., p.get_height()), 
                ha='center', 
                va='center', 
                xytext=(0, 10), 
                textcoords='offset points',
                fontsize = 14,
               )

plt.show()
```


![png](output_31_0.png)


##### Five Neighborhoods with lowest crime

> Indented block


```python
crime_neigh_low = vnc_crime_neigh.tail(5)
crime_neigh_low
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighbourhood</th>
      <th>YearBreak and Enter Commercial</th>
      <th>YearBreak and Enter Residential/Other</th>
      <th>YearMischief</th>
      <th>YearOther Theft</th>
      <th>YearTheft from Vehicle</th>
      <th>YearTheft of Bicycle</th>
      <th>YearTheft of Vehicle</th>
      <th>YearVehicle Collision or Pedestrian Struck (with Fatality)</th>
      <th>YearVehicle Collision or Pedestrian Struck (with Injury)</th>
      <th>Total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>23</th>
      <td>West Point Grey</td>
      <td>18</td>
      <td>71</td>
      <td>50</td>
      <td>11</td>
      <td>157</td>
      <td>32</td>
      <td>11</td>
      <td>0</td>
      <td>22</td>
      <td>372</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Arbutus Ridge</td>
      <td>12</td>
      <td>78</td>
      <td>49</td>
      <td>18</td>
      <td>111</td>
      <td>12</td>
      <td>12</td>
      <td>1</td>
      <td>18</td>
      <td>311</td>
    </tr>
    <tr>
      <th>17</th>
      <td>South Cambie</td>
      <td>22</td>
      <td>42</td>
      <td>41</td>
      <td>38</td>
      <td>111</td>
      <td>19</td>
      <td>8</td>
      <td>0</td>
      <td>11</td>
      <td>292</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Stanley Park</td>
      <td>6</td>
      <td>2</td>
      <td>8</td>
      <td>0</td>
      <td>109</td>
      <td>14</td>
      <td>3</td>
      <td>0</td>
      <td>12</td>
      <td>154</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Musqueam</td>
      <td>0</td>
      <td>4</td>
      <td>3</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>17</td>
    </tr>
  </tbody>
</table>
</div>




```python
per_neigh = crime_neigh_low[['Neighbourhood','Total']]

per_neigh.set_index('Neighbourhood',inplace = True)

ax = per_neigh.plot(kind='bar', figsize=(10, 6), rot=0)

ax.set_ylabel('Number of Crimes')
ax.set_xlabel('Neighbourhood')
ax.set_title('Neighbourhoods in Vancouver with the lowest crimes')

for p in ax.patches:
    ax.annotate(np.round(p.get_height(),decimals=2), 
                (p.get_x()+p.get_width()/2., p.get_height()), 
                ha='center', 
                va='center', 
                xytext=(0, 10), 
                textcoords='offset points',
                fontsize = 14,
               )

plt.show()
```


![png](output_34_0.png)


#### Borough is Vancouver with Highest Crime


```python
vnc_crime_cat = pd.pivot_table(vnc_boroughs_crime,
                               values=['Year'],
                               index=['Borough'],
                               columns=['Type'],
                               aggfunc=len,
                               fill_value=0,
                               margins=True)
vnc_crime_cat
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="10" halign="left">Year</th>
    </tr>
    <tr>
      <th>Type</th>
      <th>Break and Enter Commercial</th>
      <th>Break and Enter Residential/Other</th>
      <th>Mischief</th>
      <th>Other Theft</th>
      <th>Theft from Vehicle</th>
      <th>Theft of Bicycle</th>
      <th>Theft of Vehicle</th>
      <th>Vehicle Collision or Pedestrian Struck (with Fatality)</th>
      <th>Vehicle Collision or Pedestrian Struck (with Injury)</th>
      <th>All</th>
    </tr>
    <tr>
      <th>Borough</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Central</th>
      <td>787</td>
      <td>198</td>
      <td>2280</td>
      <td>2489</td>
      <td>6871</td>
      <td>857</td>
      <td>245</td>
      <td>1</td>
      <td>314</td>
      <td>14042</td>
    </tr>
    <tr>
      <th>East Side</th>
      <td>786</td>
      <td>1043</td>
      <td>2192</td>
      <td>1674</td>
      <td>4754</td>
      <td>678</td>
      <td>605</td>
      <td>8</td>
      <td>660</td>
      <td>12400</td>
    </tr>
    <tr>
      <th>South Vancouver</th>
      <td>49</td>
      <td>156</td>
      <td>187</td>
      <td>88</td>
      <td>483</td>
      <td>36</td>
      <td>71</td>
      <td>1</td>
      <td>111</td>
      <td>1182</td>
    </tr>
    <tr>
      <th>West Side</th>
      <td>403</td>
      <td>1000</td>
      <td>1062</td>
      <td>696</td>
      <td>2838</td>
      <td>588</td>
      <td>225</td>
      <td>3</td>
      <td>389</td>
      <td>7204</td>
    </tr>
    <tr>
      <th>All</th>
      <td>2025</td>
      <td>2397</td>
      <td>5721</td>
      <td>4947</td>
      <td>14946</td>
      <td>2159</td>
      <td>1146</td>
      <td>13</td>
      <td>1474</td>
      <td>34828</td>
    </tr>
  </tbody>
</table>
</div>




```python
vnc_crime_cat.reset_index(inplace = True)
vnc_crime_cat.columns = vnc_crime_cat.columns.map(''.join)
vnc_crime_cat.rename(columns={'YearAll':'Total',
                              'YearBreak and Enter Commercial' : 'Break and Enter Commercial',
                              'YearBreak and Enter Residential/Other' : 'Break and Enter Residential',
                              'YearMischief' : 'Mischief',
                              'YearOther Theft' : 'Other',
                              'YearTheft from Vehicle' : 'Theft from Vehicle',
                              'YearTheft of Bicycle' : 'Theft of Bicycle',
                              'YearTheft of Vehicle' : 'Theft of Vehicle',
                              'YearVehicle Collision or Pedestrian Struck (with Fatality)' : 'Vehicle Collision or Pedestrian Struck (with Fatality)',
                              'YearVehicle Collision or Pedestrian Struck (with Injury)' : 'Vehicle Collision or Pedestrian Struck (with Injury)'}, inplace=True)
# To ignore bottom All in Borough
vnc_crime_cat = vnc_crime_cat.head(4)
vnc_crime_cat
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough</th>
      <th>Break and Enter Commercial</th>
      <th>Break and Enter Residential</th>
      <th>Mischief</th>
      <th>Other</th>
      <th>Theft from Vehicle</th>
      <th>Theft of Bicycle</th>
      <th>Theft of Vehicle</th>
      <th>Vehicle Collision or Pedestrian Struck (with Fatality)</th>
      <th>Vehicle Collision or Pedestrian Struck (with Injury)</th>
      <th>Total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Central</td>
      <td>787</td>
      <td>198</td>
      <td>2280</td>
      <td>2489</td>
      <td>6871</td>
      <td>857</td>
      <td>245</td>
      <td>1</td>
      <td>314</td>
      <td>14042</td>
    </tr>
    <tr>
      <th>1</th>
      <td>East Side</td>
      <td>786</td>
      <td>1043</td>
      <td>2192</td>
      <td>1674</td>
      <td>4754</td>
      <td>678</td>
      <td>605</td>
      <td>8</td>
      <td>660</td>
      <td>12400</td>
    </tr>
    <tr>
      <th>2</th>
      <td>South Vancouver</td>
      <td>49</td>
      <td>156</td>
      <td>187</td>
      <td>88</td>
      <td>483</td>
      <td>36</td>
      <td>71</td>
      <td>1</td>
      <td>111</td>
      <td>1182</td>
    </tr>
    <tr>
      <th>3</th>
      <td>West Side</td>
      <td>403</td>
      <td>1000</td>
      <td>1062</td>
      <td>696</td>
      <td>2838</td>
      <td>588</td>
      <td>225</td>
      <td>3</td>
      <td>389</td>
      <td>7204</td>
    </tr>
  </tbody>
</table>
</div>




```python
per_borough = vnc_crime_cat[['Borough','Total']]

per_borough.set_index('Borough',inplace = True)

ax = per_borough.plot(kind='bar', figsize=(10, 6), rot=0)

ax.set_ylabel('Number of Crimes')
ax.set_xlabel('Borough')
ax.set_title('Boroughs in Vancouver with the Highest crimes')

for p in ax.patches:
    ax.annotate(np.round(p.get_height(),decimals=2), 
                (p.get_x()+p.get_width()/2., p.get_height()), 
                ha='center', 
                va='center', 
                xytext=(0, 10), 
                textcoords='offset points',
                fontsize = 14,
               )

plt.show()
```


![png](output_38_0.png)


### Based on exploratory data analysis it is clear that South Vancouver has the lowest crimes

##### Since South Vancouver has very little number of neighborhoods and opening a commercial establishment would not be viable, we can choose the next borough with lowest crime which is **West Side**.

#### Different types of crimes recorded in the West Side Borough

#### West side was chosen because crime type Break and enter Commercial is also low amongst other crimes types which makes West Side ideal destination for opening of commercial establishments


```python
vnc_ws_df = vnc_crime_cat[vnc_crime_cat['Borough'] == 'West Side']

vnc_ws_df = vnc_ws_df.sort_values(['Total'], ascending = True, axis = 0)

vnc_ws = vnc_ws_df[['Borough','Theft of Vehicle', 'Break and Enter Commercial','Break and Enter Residential','Mischief','Other',
                 'Theft from Vehicle','Vehicle Collision or Pedestrian Struck (with Fatality)','Theft of Bicycle',
                 'Vehicle Collision or Pedestrian Struck (with Injury)']]


vnc_ws.set_index('Borough',inplace = True)

ax = vnc_ws.plot(kind='bar', figsize=(10, 6), rot=0)

ax.set_ylabel('Number of Crimes')
ax.set_xlabel('Borough')
ax.set_title('Different Kind of Crimes in West Side Borough')

for p in ax.patches:
    ax.annotate(np.round(p.get_height(),decimals=3), 
                (p.get_x()+p.get_width()/3., p.get_height()), 
                ha='center', 
                va='center', 
                xytext=(5, 10), 
                textcoords='offset points',
                fontsize = 14
               )
    ax.legend(loc='upper left', bbox_to_anchor=(1.00, 0.5))

plt.show()
```


![png](output_41_0.png)


### **Part 3**: Creating a new consolidated dataset of the Neighborhoods, along with their boroughs, crime data and the respective Neighbourhood's co-ordinates.<a name="part3"></a>: 

#### This data will be fetched using OpenCage Geocoder to find the safest borough and explore the neighbourhood by plotting it on maps using Folium and perform exploratory data analysis.

##### Restricting the rows in the data frame to only those with West side as Borough


```python
vnc_ws_neigh = vnc_boroughs_crime

#vnc_ws_neigh.drop(['Type','Year', 'Month', 'Day', 'Hour'], axis = 1, inplace = True)
vnc_ws_neigh = vnc_ws_neigh[vnc_ws_neigh['Borough'] == 'West Side']
vnc_ws_neigh.reset_index(inplace=True, drop=True)

print('Number of Neighbourhoods in West Side Borough', len(vnc_ws_neigh['Neighbourhood'].unique()))

vnc_ws_neigh['Neighbourhood'].unique()
```

    Number of Neighbourhoods in West Side Borough 10





    array(['Shaughnessy', 'Fairview', 'Oakridge', 'Marpole', 'Kitsilano',
           'Kerrisdale', 'West Point Grey', 'Arbutus Ridge', 'South Cambie',
           'Dunbar-Southlands'], dtype=object)



##### Creating a new Data frame with Lat, Lng being fetched from OpenCage geocoder


```python
Latitude = []
Longitude = []
Borough = []
Neighbourhood = vnc_ws_neigh['Neighbourhood'].unique()



key = '830323b5ca694362904814ff0a11b803'
geocoder = OpenCageGeocode(key)

for i in range(len(Neighbourhood)):
    address = '{}, Vancouver, BC, Canada'.format(Neighbourhood[i])
    location = geocoder.geocode(address)
    Latitude.append(location[0]['geometry']['lat'])
    Longitude.append(location[0]['geometry']['lng'])
    Borough.append('West Side')
print(Latitude, Longitude)

#print('The geograpical coordinate of Vancouver City are {}, {}.'.format(latitude, longitude))
```

    [49.2518626, 49.2641128, 49.2308288, 49.2092233, 49.2694099, 49.2346728, 49.2644843, 49.2409677, 49.2466847, 49.2534601] [-123.1380226, -123.1268352, -123.1311342, -123.1361495, -123.155267, -123.1553893, -123.1854326, -123.1670008, -123.120915, -123.1850439]


#### Glimpse of the new Data Frame with Neighborhoods in West Side Borough of Vancoouver along with centroid of their co-ordinates


```python
ws_neig_dict = {'Neighbourhood': Neighbourhood,'Borough':Borough,'Latitude': Latitude,'Longitude':Longitude}
ws_neig_geo = pd.DataFrame(data=ws_neig_dict, columns=['Neighbourhood', 'Borough', 'Latitude', 'Longitude'], index=None)

ws_neig_geo
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighbourhood</th>
      <th>Borough</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Shaughnessy</td>
      <td>West Side</td>
      <td>49.251863</td>
      <td>-123.138023</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Fairview</td>
      <td>West Side</td>
      <td>49.264113</td>
      <td>-123.126835</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Oakridge</td>
      <td>West Side</td>
      <td>49.230829</td>
      <td>-123.131134</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Marpole</td>
      <td>West Side</td>
      <td>49.209223</td>
      <td>-123.136150</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Kitsilano</td>
      <td>West Side</td>
      <td>49.269410</td>
      <td>-123.155267</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Kerrisdale</td>
      <td>West Side</td>
      <td>49.234673</td>
      <td>-123.155389</td>
    </tr>
    <tr>
      <th>6</th>
      <td>West Point Grey</td>
      <td>West Side</td>
      <td>49.264484</td>
      <td>-123.185433</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Arbutus Ridge</td>
      <td>West Side</td>
      <td>49.240968</td>
      <td>-123.167001</td>
    </tr>
    <tr>
      <th>8</th>
      <td>South Cambie</td>
      <td>West Side</td>
      <td>49.246685</td>
      <td>-123.120915</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Dunbar-Southlands</td>
      <td>West Side</td>
      <td>49.253460</td>
      <td>-123.185044</td>
    </tr>
  </tbody>
</table>
</div>



#### Fetching the Geographical co-ordiantes of Vancouver to plot on Map


```python
address = 'Vancouver, BC, Canada'

location = geocoder.geocode(address)
latitude = location[0]['geometry']['lat']
longitude = location[0]['geometry']['lng']

print('The geograpical coordinate of Vancouver, Canada are {}, {}.'.format(latitude, longitude))
```

    The geograpical coordinate of Vancouver, Canada are 49.2608724, -123.1139529.


#### Using Folium to plot Vancouver City's West Side Borough and it's Neighborhoods


```python
van_map = folium.Map(location=[latitude, longitude], zoom_start=12)

# add markers to map
for lat, lng, borough, neighborhood in zip(ws_neig_geo['Latitude'], ws_neig_geo['Longitude'], ws_neig_geo['Borough'], ws_neig_geo['Neighbourhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='red',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(van_map)  
    
van_map
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><span style="color:#565656">Make this Notebook Trusted to load map: File -> Trust Notebook</span><iframe src="about:blank" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" data-html=PCFET0NUWVBFIGh0bWw+CjxoZWFkPiAgICAKICAgIDxtZXRhIGh0dHAtZXF1aXY9ImNvbnRlbnQtdHlwZSIgY29udGVudD0idGV4dC9odG1sOyBjaGFyc2V0PVVURi04IiAvPgogICAgPHNjcmlwdD5MX1BSRUZFUl9DQU5WQVMgPSBmYWxzZTsgTF9OT19UT1VDSCA9IGZhbHNlOyBMX0RJU0FCTEVfM0QgPSBmYWxzZTs8L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmpzIj48L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2FqYXguZ29vZ2xlYXBpcy5jb20vYWpheC9saWJzL2pxdWVyeS8xLjExLjEvanF1ZXJ5Lm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvanMvYm9vdHN0cmFwLm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9jZG5qcy5jbG91ZGZsYXJlLmNvbS9hamF4L2xpYnMvTGVhZmxldC5hd2Vzb21lLW1hcmtlcnMvMi4wLjIvbGVhZmxldC5hd2Vzb21lLW1hcmtlcnMuanMiPjwvc2NyaXB0PgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9jc3MvYm9vdHN0cmFwLm1pbi5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvY3NzL2Jvb3RzdHJhcC10aGVtZS5taW4uY3NzIi8+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vbWF4Y2RuLmJvb3RzdHJhcGNkbi5jb20vZm9udC1hd2Vzb21lLzQuNi4zL2Nzcy9mb250LWF3ZXNvbWUubWluLmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2NkbmpzLmNsb3VkZmxhcmUuY29tL2FqYXgvbGlicy9MZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy8yLjAuMi9sZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9yYXdnaXQuY29tL3B5dGhvbi12aXN1YWxpemF0aW9uL2ZvbGl1bS9tYXN0ZXIvZm9saXVtL3RlbXBsYXRlcy9sZWFmbGV0LmF3ZXNvbWUucm90YXRlLmNzcyIvPgogICAgPHN0eWxlPmh0bWwsIGJvZHkge3dpZHRoOiAxMDAlO2hlaWdodDogMTAwJTttYXJnaW46IDA7cGFkZGluZzogMDt9PC9zdHlsZT4KICAgIDxzdHlsZT4jbWFwIHtwb3NpdGlvbjphYnNvbHV0ZTt0b3A6MDtib3R0b206MDtyaWdodDowO2xlZnQ6MDt9PC9zdHlsZT4KICAgIAogICAgICAgICAgICA8c3R5bGU+ICNtYXBfZTY1YTNmZmVmNTBmNGFmYTkxNWU2Mjg2MWFkODk1MDQgewogICAgICAgICAgICAgICAgcG9zaXRpb24gOiByZWxhdGl2ZTsKICAgICAgICAgICAgICAgIHdpZHRoIDogMTAwLjAlOwogICAgICAgICAgICAgICAgaGVpZ2h0OiAxMDAuMCU7CiAgICAgICAgICAgICAgICBsZWZ0OiAwLjAlOwogICAgICAgICAgICAgICAgdG9wOiAwLjAlOwogICAgICAgICAgICAgICAgfQogICAgICAgICAgICA8L3N0eWxlPgogICAgICAgIAo8L2hlYWQ+Cjxib2R5PiAgICAKICAgIAogICAgICAgICAgICA8ZGl2IGNsYXNzPSJmb2xpdW0tbWFwIiBpZD0ibWFwX2U2NWEzZmZlZjUwZjRhZmE5MTVlNjI4NjFhZDg5NTA0IiA+PC9kaXY+CiAgICAgICAgCjwvYm9keT4KPHNjcmlwdD4gICAgCiAgICAKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGJvdW5kcyA9IG51bGw7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgdmFyIG1hcF9lNjVhM2ZmZWY1MGY0YWZhOTE1ZTYyODYxYWQ4OTUwNCA9IEwubWFwKAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJ21hcF9lNjVhM2ZmZWY1MGY0YWZhOTE1ZTYyODYxYWQ4OTUwNCcsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB7Y2VudGVyOiBbNDkuMjYwODcyNCwtMTIzLjExMzk1MjldLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgem9vbTogMTIsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBtYXhCb3VuZHM6IGJvdW5kcywKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGxheWVyczogW10sCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB3b3JsZENvcHlKdW1wOiBmYWxzZSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGNyczogTC5DUlMuRVBTRzM4NTcKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgfSk7CiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciB0aWxlX2xheWVyXzQ4ZGIxOTgwZmJjZjRlN2FiMGMyNjc4ODMyZjhiODgxID0gTC50aWxlTGF5ZXIoCiAgICAgICAgICAgICAgICAnaHR0cHM6Ly97c30udGlsZS5vcGVuc3RyZWV0bWFwLm9yZy97en0ve3h9L3t5fS5wbmcnLAogICAgICAgICAgICAgICAgewogICJhdHRyaWJ1dGlvbiI6IG51bGwsCiAgImRldGVjdFJldGluYSI6IGZhbHNlLAogICJtYXhab29tIjogMTgsCiAgIm1pblpvb20iOiAxLAogICJub1dyYXAiOiBmYWxzZSwKICAic3ViZG9tYWlucyI6ICJhYmMiCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2U2NWEzZmZlZjUwZjRhZmE5MTVlNjI4NjFhZDg5NTA0KTsKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl84ZmMwMjBkNjFjNDI0NDc5ODRhZDM4YzcxOTIwMWVmNyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQ5LjI1MTg2MjYsLTEyMy4xMzgwMjI2XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInJlZCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZTY1YTNmZmVmNTBmNGFmYTkxNWU2Mjg2MWFkODk1MDQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZTBiZGMwM2RmNGE0NGE3ZmIxYTcwZTUxMzFkOWU0MmYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMjEyMGQzNjU4ZmZkNGZmMmFhNWVhNTBiMTU0ZjkwN2IgPSAkKCc8ZGl2IGlkPSJodG1sXzIxMjBkMzY1OGZmZDRmZjJhYTVlYTUwYjE1NGY5MDdiIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5TaGF1Z2huZXNzeSwgV2VzdCBTaWRlPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9lMGJkYzAzZGY0YTQ0YTdmYjFhNzBlNTEzMWQ5ZTQyZi5zZXRDb250ZW50KGh0bWxfMjEyMGQzNjU4ZmZkNGZmMmFhNWVhNTBiMTU0ZjkwN2IpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOGZjMDIwZDYxYzQyNDQ3OTg0YWQzOGM3MTkyMDFlZjcuYmluZFBvcHVwKHBvcHVwX2UwYmRjMDNkZjRhNDRhN2ZiMWE3MGU1MTMxZDllNDJmKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzZiMDRjYzkxZTY3OTRjYjM5NTE4OTNlMTQ0MTI1ZDNjID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDkuMjY0MTEyOCwtMTIzLjEyNjgzNTJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAicmVkIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9lNjVhM2ZmZWY1MGY0YWZhOTE1ZTYyODYxYWQ4OTUwNCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kOThmNGYwZWNmZjc0Zjk3OTFmMTM1NjlkY2IzZDkwZSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9jZDNiNDk3NDE2NWU0OGMzOTA3NTdhZWMwMjhlNTc5NyA9ICQoJzxkaXYgaWQ9Imh0bWxfY2QzYjQ5NzQxNjVlNDhjMzkwNzU3YWVjMDI4ZTU3OTciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkZhaXJ2aWV3LCBXZXN0IFNpZGU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2Q5OGY0ZjBlY2ZmNzRmOTc5MWYxMzU2OWRjYjNkOTBlLnNldENvbnRlbnQoaHRtbF9jZDNiNDk3NDE2NWU0OGMzOTA3NTdhZWMwMjhlNTc5Nyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl82YjA0Y2M5MWU2Nzk0Y2IzOTUxODkzZTE0NDEyNWQzYy5iaW5kUG9wdXAocG9wdXBfZDk4ZjRmMGVjZmY3NGY5NzkxZjEzNTY5ZGNiM2Q5MGUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMjE3YzBkMjY5MzUyNGRiNTg2ZmE2YTI0NGUxYjgyZTQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0OS4yMzA4Mjg4LC0xMjMuMTMxMTM0Ml0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJyZWQiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2U2NWEzZmZlZjUwZjRhZmE5MTVlNjI4NjFhZDg5NTA0KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzE1NDk1ZmRlYzllMTQ2YjhhOWE4ZTVkY2Q2ZjUzZTNhID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzkxYjU4YmNmMTVmYzQzMzBhZjI1YTk3MDdhNWI1Yzg0ID0gJCgnPGRpdiBpZD0iaHRtbF85MWI1OGJjZjE1ZmM0MzMwYWYyNWE5NzA3YTViNWM4NCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+T2FrcmlkZ2UsIFdlc3QgU2lkZTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMTU0OTVmZGVjOWUxNDZiOGE5YThlNWRjZDZmNTNlM2Euc2V0Q29udGVudChodG1sXzkxYjU4YmNmMTVmYzQzMzBhZjI1YTk3MDdhNWI1Yzg0KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzIxN2MwZDI2OTM1MjRkYjU4NmZhNmEyNDRlMWI4MmU0LmJpbmRQb3B1cChwb3B1cF8xNTQ5NWZkZWM5ZTE0NmI4YTlhOGU1ZGNkNmY1M2UzYSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jOTk2NTZlMTViOTE0ZDZmODU2NDQyM2E3NjNjMDgzYyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQ5LjIwOTIyMzMsLTEyMy4xMzYxNDk1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInJlZCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZTY1YTNmZmVmNTBmNGFmYTkxNWU2Mjg2MWFkODk1MDQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfODczMjliNGJjZTkyNDkwZWIxODI3MmY3NTViNDk5OTEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYjc2NGZiM2E4ZmY0NDgwNzgxZTYxOGQ5Y2E3YmYyMGYgPSAkKCc8ZGl2IGlkPSJodG1sX2I3NjRmYjNhOGZmNDQ4MDc4MWU2MThkOWNhN2JmMjBmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5NYXJwb2xlLCBXZXN0IFNpZGU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzg3MzI5YjRiY2U5MjQ5MGViMTgyNzJmNzU1YjQ5OTkxLnNldENvbnRlbnQoaHRtbF9iNzY0ZmIzYThmZjQ0ODA3ODFlNjE4ZDljYTdiZjIwZik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jOTk2NTZlMTViOTE0ZDZmODU2NDQyM2E3NjNjMDgzYy5iaW5kUG9wdXAocG9wdXBfODczMjliNGJjZTkyNDkwZWIxODI3MmY3NTViNDk5OTEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNzNiMDkwZGUwNmRjNDIwNjkzNDdjNmZmNWNhZDEzYWQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0OS4yNjk0MDk5LC0xMjMuMTU1MjY3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInJlZCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZTY1YTNmZmVmNTBmNGFmYTkxNWU2Mjg2MWFkODk1MDQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNGI0ZTlkMmNjZTBiNGFhZmFjNGM5ODViNWQ1ZmJjZTIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZmUxMTUyYjdlZDNlNGVhNWJkZWRjMjMzYWY1MzI2MzQgPSAkKCc8ZGl2IGlkPSJodG1sX2ZlMTE1MmI3ZWQzZTRlYTViZGVkYzIzM2FmNTMyNjM0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5LaXRzaWxhbm8sIFdlc3QgU2lkZTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNGI0ZTlkMmNjZTBiNGFhZmFjNGM5ODViNWQ1ZmJjZTIuc2V0Q29udGVudChodG1sX2ZlMTE1MmI3ZWQzZTRlYTViZGVkYzIzM2FmNTMyNjM0KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzczYjA5MGRlMDZkYzQyMDY5MzQ3YzZmZjVjYWQxM2FkLmJpbmRQb3B1cChwb3B1cF80YjRlOWQyY2NlMGI0YWFmYWM0Yzk4NWI1ZDVmYmNlMik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl83YTIyNDY0MjMyNmY0YjBjOGM1MDE1N2UzYmM4ODgyYyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQ5LjIzNDY3MjgsLTEyMy4xNTUzODkzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInJlZCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZTY1YTNmZmVmNTBmNGFmYTkxNWU2Mjg2MWFkODk1MDQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNjBiYTIyZTE0ODU1NGMxMWJmNDZkMjQxN2ZkY2U2YmYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfY2RhZjM4Nzg2MmRlNGUyZjgwZjhlYjYwYWU5YzYyOGIgPSAkKCc8ZGl2IGlkPSJodG1sX2NkYWYzODc4NjJkZTRlMmY4MGY4ZWI2MGFlOWM2MjhiIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5LZXJyaXNkYWxlLCBXZXN0IFNpZGU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzYwYmEyMmUxNDg1NTRjMTFiZjQ2ZDI0MTdmZGNlNmJmLnNldENvbnRlbnQoaHRtbF9jZGFmMzg3ODYyZGU0ZTJmODBmOGViNjBhZTljNjI4Yik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl83YTIyNDY0MjMyNmY0YjBjOGM1MDE1N2UzYmM4ODgyYy5iaW5kUG9wdXAocG9wdXBfNjBiYTIyZTE0ODU1NGMxMWJmNDZkMjQxN2ZkY2U2YmYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMmI2NWJiOTFlZWIyNGMyM2FjNmE4ZTc5NGIwMGY2ZmIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0OS4yNjQ0ODQzLC0xMjMuMTg1NDMyNl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJyZWQiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2U2NWEzZmZlZjUwZjRhZmE5MTVlNjI4NjFhZDg5NTA0KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzc0Mjg3ODk5ZGE1NzQ0ZTk5NTUwOWU1NjIzMzcyNjY0ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzNjOTIzODBkNTFlODRlNDg5NjNkNTVmMjY1YzYzYjM2ID0gJCgnPGRpdiBpZD0iaHRtbF8zYzkyMzgwZDUxZTg0ZTQ4OTYzZDU1ZjI2NWM2M2IzNiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+V2VzdCBQb2ludCBHcmV5LCBXZXN0IFNpZGU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzc0Mjg3ODk5ZGE1NzQ0ZTk5NTUwOWU1NjIzMzcyNjY0LnNldENvbnRlbnQoaHRtbF8zYzkyMzgwZDUxZTg0ZTQ4OTYzZDU1ZjI2NWM2M2IzNik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8yYjY1YmI5MWVlYjI0YzIzYWM2YThlNzk0YjAwZjZmYi5iaW5kUG9wdXAocG9wdXBfNzQyODc4OTlkYTU3NDRlOTk1NTA5ZTU2MjMzNzI2NjQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMGFiMGFhNTgyNjNmNDE3YTk0NDZjNjM4NWQ1NjM5YjQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0OS4yNDA5Njc3LC0xMjMuMTY3MDAwOF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJyZWQiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2U2NWEzZmZlZjUwZjRhZmE5MTVlNjI4NjFhZDg5NTA0KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2NhMjc3ZDY5Zjk4YTQzOThiNjYyNDY2ZWNjYzdkNzgwID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2JiZWNjMmQxM2I5YTQwM2ZiMDBkMDQ1MjM0YWM5MTdlID0gJCgnPGRpdiBpZD0iaHRtbF9iYmVjYzJkMTNiOWE0MDNmYjAwZDA0NTIzNGFjOTE3ZSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QXJidXR1cyBSaWRnZSwgV2VzdCBTaWRlPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9jYTI3N2Q2OWY5OGE0Mzk4YjY2MjQ2NmVjY2M3ZDc4MC5zZXRDb250ZW50KGh0bWxfYmJlY2MyZDEzYjlhNDAzZmIwMGQwNDUyMzRhYzkxN2UpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMGFiMGFhNTgyNjNmNDE3YTk0NDZjNjM4NWQ1NjM5YjQuYmluZFBvcHVwKHBvcHVwX2NhMjc3ZDY5Zjk4YTQzOThiNjYyNDY2ZWNjYzdkNzgwKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2E0ZDNjOGEyZGQwOTQ1MDBhNjJmOGI3NWY2ZmU1NGNkID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDkuMjQ2Njg0NywtMTIzLjEyMDkxNV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJyZWQiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2U2NWEzZmZlZjUwZjRhZmE5MTVlNjI4NjFhZDg5NTA0KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzAzYWU5NDI1ZWU3YTQ1OTdiYTZhMzEzYWFjNDhlYTc5ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzRlMTY1ZjBlNjgwNjQwNmZhYTg2MWJlNDgwMTEzODMyID0gJCgnPGRpdiBpZD0iaHRtbF80ZTE2NWYwZTY4MDY0MDZmYWE4NjFiZTQ4MDExMzgzMiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U291dGggQ2FtYmllLCBXZXN0IFNpZGU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzAzYWU5NDI1ZWU3YTQ1OTdiYTZhMzEzYWFjNDhlYTc5LnNldENvbnRlbnQoaHRtbF80ZTE2NWYwZTY4MDY0MDZmYWE4NjFiZTQ4MDExMzgzMik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hNGQzYzhhMmRkMDk0NTAwYTYyZjhiNzVmNmZlNTRjZC5iaW5kUG9wdXAocG9wdXBfMDNhZTk0MjVlZTdhNDU5N2JhNmEzMTNhYWM0OGVhNzkpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMTkwMzg2N2EyNGU0NGUzNjg2NWZjNzExOGI0ZmViZTEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0OS4yNTM0NjAxLC0xMjMuMTg1MDQzOV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJyZWQiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2U2NWEzZmZlZjUwZjRhZmE5MTVlNjI4NjFhZDg5NTA0KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzk2NWM2MGUzMjJkZjQ0YWM5NjRiODdhNjgwNTgxNzNmID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzRiODJkMTUzOTUxNTRmOGI5Mzc5NWIzMGUzNjE0ZGEyID0gJCgnPGRpdiBpZD0iaHRtbF80YjgyZDE1Mzk1MTU0ZjhiOTM3OTViMzBlMzYxNGRhMiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RHVuYmFyLVNvdXRobGFuZHMsIFdlc3QgU2lkZTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOTY1YzYwZTMyMmRmNDRhYzk2NGI4N2E2ODA1ODE3M2Yuc2V0Q29udGVudChodG1sXzRiODJkMTUzOTUxNTRmOGI5Mzc5NWIzMGUzNjE0ZGEyKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzE5MDM4NjdhMjRlNDRlMzY4NjVmYzcxMThiNGZlYmUxLmJpbmRQb3B1cChwb3B1cF85NjVjNjBlMzIyZGY0NGFjOTY0Yjg3YTY4MDU4MTczZik7CgogICAgICAgICAgICAKICAgICAgICAKPC9zY3JpcHQ+ onload="this.contentDocument.open();this.contentDocument.write(atob(this.getAttribute('data-html')));this.contentDocument.close();" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



### **Part 4**: Creating a new consolidated dataset of the Neighborhoods, boroughs, and the most common venues and the respective Neighbourhood along with co-ordinates.<a name="part4"></a>: 
##### This data will be fetched using Four Square API to explore the neighbourhood venues and to apply machine learning algorithm to cluster the neighbourhoods and present the findings by plotting it on maps using Folium.

#### Setting Up Foursquare Credentials

> Indented block





```python
#Four Square Credentials

CLIENT_ID = 'XVY0YGK3DX5QGHMN2TGSK2EWA55P3JNPIVC5QVW5SGIGUI2L'
CLIENT_SECRET = 'T53Z3HT4W5DVALRIPBK2DPD4NFOCISMUTMNBLNW13KEJTAIJ'
VERSION = '20191101'
LIMIT = 100

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)
```

    Your credentails:
    CLIENT_ID: XVY0YGK3DX5QGHMN2TGSK2EWA55P3JNPIVC5QVW5SGIGUI2L
    CLIENT_SECRET:T53Z3HT4W5DVALRIPBK2DPD4NFOCISMUTMNBLNW13KEJTAIJ


#### Defining a function to fetch top 10 venues around a given neighborhood


```python
def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighbourhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Category']
    
    return(nearby_venues)
```

#### Generating Venues


```python
vnc_ws_venues = getNearbyVenues(names=ws_neig_geo['Neighbourhood'],
                                   latitudes=ws_neig_geo['Latitude'],
                                   longitudes=ws_neig_geo['Longitude']
                                  )
```

    Shaughnessy
    Fairview
    Oakridge
    Marpole
    Kitsilano
    Kerrisdale
    West Point Grey
    Arbutus Ridge
    South Cambie
    Dunbar-Southlands


#### Data frame containing venues for each neighborhood in West Side


```python
print(vnc_ws_venues.shape)
vnc_ws_venues.head()
```

    (220, 5)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighbourhood</th>
      <th>Neighborhood Latitude</th>
      <th>Neighborhood Longitude</th>
      <th>Venue</th>
      <th>Venue Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Shaughnessy</td>
      <td>49.251863</td>
      <td>-123.138023</td>
      <td>Angus Park</td>
      <td>Park</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Shaughnessy</td>
      <td>49.251863</td>
      <td>-123.138023</td>
      <td>Crepe &amp; Cafe</td>
      <td>French Restaurant</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Fairview</td>
      <td>49.264113</td>
      <td>-123.126835</td>
      <td>Gyu-Kaku Japanese BBQ</td>
      <td>BBQ Joint</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Fairview</td>
      <td>49.264113</td>
      <td>-123.126835</td>
      <td>CRESCENT nail and spa</td>
      <td>Nail Salon</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Fairview</td>
      <td>49.264113</td>
      <td>-123.126835</td>
      <td>Charleson Park</td>
      <td>Park</td>
    </tr>
  </tbody>
</table>
</div>



#### Venue Count per neighborhood


```python
vnc_ws_venues.groupby('Neighbourhood').count().drop(['Neighborhood Latitude','Neighborhood Longitude','Venue Category'], axis = 1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Venue</th>
    </tr>
    <tr>
      <th>Neighbourhood</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Arbutus Ridge</th>
      <td>4</td>
    </tr>
    <tr>
      <th>Dunbar-Southlands</th>
      <td>5</td>
    </tr>
    <tr>
      <th>Fairview</th>
      <td>26</td>
    </tr>
    <tr>
      <th>Kerrisdale</th>
      <td>40</td>
    </tr>
    <tr>
      <th>Kitsilano</th>
      <td>46</td>
    </tr>
    <tr>
      <th>Marpole</th>
      <td>29</td>
    </tr>
    <tr>
      <th>Oakridge</th>
      <td>8</td>
    </tr>
    <tr>
      <th>Shaughnessy</th>
      <td>2</td>
    </tr>
    <tr>
      <th>South Cambie</th>
      <td>17</td>
    </tr>
    <tr>
      <th>West Point Grey</th>
      <td>43</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('There are {} uniques categories.'.format(len(vnc_ws_venues['Venue Category'].unique())))
```

    There are 87 uniques categories.


### Modelling

##### One Hot Encoding to Analyze Each Neighborhood


```python
# one hot encoding
vnc_onehot = pd.get_dummies(vnc_ws_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
vnc_onehot['Neighbourhood'] = vnc_ws_venues['Neighbourhood'] 

# move neighborhood column to the first column
fixed_columns = [vnc_onehot.columns[-1]] + list(vnc_onehot.columns[:-1])
vnc_onehot = vnc_onehot[fixed_columns]

vnc_onehot.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighbourhood</th>
      <th>American Restaurant</th>
      <th>Asian Restaurant</th>
      <th>BBQ Joint</th>
      <th>Bakery</th>
      <th>Bank</th>
      <th>Bar</th>
      <th>Beach</th>
      <th>Bistro</th>
      <th>Bookstore</th>
      <th>...</th>
      <th>Taiwanese Restaurant</th>
      <th>Tea Room</th>
      <th>Tennis Court</th>
      <th>Thai Restaurant</th>
      <th>Thrift / Vintage Store</th>
      <th>Vegetarian / Vegan Restaurant</th>
      <th>Video Store</th>
      <th>Vietnamese Restaurant</th>
      <th>Wine Shop</th>
      <th>Yoga Studio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Shaughnessy</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Shaughnessy</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Fairview</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Fairview</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Fairview</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 88 columns</p>
</div>




```python
vnc_onehot.shape
```




    (220, 88)




```python
vnc_ws_grouped = vnc_onehot.groupby('Neighbourhood').mean().reset_index()
vnc_ws_grouped
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighbourhood</th>
      <th>American Restaurant</th>
      <th>Asian Restaurant</th>
      <th>BBQ Joint</th>
      <th>Bakery</th>
      <th>Bank</th>
      <th>Bar</th>
      <th>Beach</th>
      <th>Bistro</th>
      <th>Bookstore</th>
      <th>...</th>
      <th>Taiwanese Restaurant</th>
      <th>Tea Room</th>
      <th>Tennis Court</th>
      <th>Thai Restaurant</th>
      <th>Thrift / Vintage Store</th>
      <th>Vegetarian / Vegan Restaurant</th>
      <th>Video Store</th>
      <th>Vietnamese Restaurant</th>
      <th>Wine Shop</th>
      <th>Yoga Studio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Arbutus Ridge</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.250000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Dunbar-Southlands</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Fairview</td>
      <td>0.000000</td>
      <td>0.076923</td>
      <td>0.038462</td>
      <td>0.000000</td>
      <td>0.038462</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.038462</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Kerrisdale</td>
      <td>0.000000</td>
      <td>0.025000</td>
      <td>0.000000</td>
      <td>0.025000</td>
      <td>0.050000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.025</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.050000</td>
      <td>0.000000</td>
      <td>0.025000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.025000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Kitsilano</td>
      <td>0.043478</td>
      <td>0.021739</td>
      <td>0.000000</td>
      <td>0.065217</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.021739</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.043478</td>
      <td>0.021739</td>
      <td>0.043478</td>
      <td>0.000000</td>
      <td>0.021739</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.021739</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Marpole</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.034483</td>
      <td>0.034483</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.034483</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.034483</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.034483</td>
      <td>0.034483</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Oakridge</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Shaughnessy</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>South Cambie</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.058824</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.058824</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>West Point Grey</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.023256</td>
      <td>0.023256</td>
      <td>0.023256</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.046512</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.023256</td>
      <td>0.046512</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.023256</td>
      <td>0.023256</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 88 columns</p>
</div>




```python
vnc_ws_grouped.shape
```




    (10, 88)



#### Top 5 most common venues across neighborhoods


```python
num_top_venues = 5

for hood in vnc_ws_grouped['Neighbourhood']:
    print("----"+hood+"----")
    temp = vnc_ws_grouped[vnc_ws_grouped['Neighbourhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')
```

    ----Arbutus Ridge----
                venue  freq
    0  Nightlife Spot  0.25
    1          Bakery  0.25
    2             Spa  0.25
    3   Grocery Store  0.25
    4            Park  0.00
    
    
    ----Dunbar-Southlands----
                    venue  freq
    0  Italian Restaurant   0.2
    1         Coffee Shop   0.2
    2   Indian Restaurant   0.2
    3      Ice Cream Shop   0.2
    4    Sushi Restaurant   0.2
    
    
    ----Fairview----
                     venue  freq
    0          Coffee Shop  0.15
    1     Asian Restaurant  0.08
    2                 Park  0.08
    3  Japanese Restaurant  0.04
    4         Camera Store  0.04
    
    
    ----Kerrisdale----
                    venue  freq
    0         Coffee Shop  0.10
    1  Chinese Restaurant  0.08
    2            Pharmacy  0.05
    3     Bubble Tea Shop  0.05
    4            Tea Room  0.05
    
    
    ----Kitsilano----
                     venue  freq
    0               Bakery  0.07
    1  American Restaurant  0.04
    2          Coffee Shop  0.04
    3      Thai Restaurant  0.04
    4             Tea Room  0.04
    
    
    ----Marpole----
                     venue  freq
    0     Sushi Restaurant  0.10
    1          Pizza Place  0.07
    2       Sandwich Place  0.07
    3   Chinese Restaurant  0.07
    4  Japanese Restaurant  0.03
    
    
    ----Oakridge----
                       venue  freq
    0       Sushi Restaurant  0.12
    1  Vietnamese Restaurant  0.12
    2   Fast Food Restaurant  0.12
    3     Israeli Restaurant  0.12
    4      Convenience Store  0.12
    
    
    ----Shaughnessy----
                     venue  freq
    0    French Restaurant   0.5
    1                 Park   0.5
    2  American Restaurant   0.0
    3            Juice Bar   0.0
    4                  Pub   0.0
    
    
    ----South Cambie----
                  venue  freq
    0       Coffee Shop  0.29
    1      Liquor Store  0.06
    2  Sushi Restaurant  0.06
    3     Shopping Mall  0.06
    4     Grocery Store  0.06
    
    
    ----West Point Grey----
                     venue  freq
    0          Coffee Shop  0.09
    1  Japanese Restaurant  0.07
    2                 Café  0.07
    3     Sushi Restaurant  0.07
    4            Bookstore  0.05
    
    


#### Now let's create the new dataframe and display the top 10 venues for each neighborhood.


```python
def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]
```


```python
num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighbourhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighbourhood'] = vnc_ws_grouped['Neighbourhood']

for ind in np.arange(vnc_ws_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(vnc_ws_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighbourhood</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Arbutus Ridge</td>
      <td>Spa</td>
      <td>Bakery</td>
      <td>Grocery Store</td>
      <td>Nightlife Spot</td>
      <td>Yoga Studio</td>
      <td>Deli / Bodega</td>
      <td>Dessert Shop</td>
      <td>Dim Sum Restaurant</td>
      <td>Diner</td>
      <td>Falafel Restaurant</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Dunbar-Southlands</td>
      <td>Coffee Shop</td>
      <td>Ice Cream Shop</td>
      <td>Sushi Restaurant</td>
      <td>Indian Restaurant</td>
      <td>Italian Restaurant</td>
      <td>Gym Pool</td>
      <td>Gym / Fitness Center</td>
      <td>Cosmetics Shop</td>
      <td>Deli / Bodega</td>
      <td>Dessert Shop</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Fairview</td>
      <td>Coffee Shop</td>
      <td>Park</td>
      <td>Asian Restaurant</td>
      <td>Korean Restaurant</td>
      <td>Pharmacy</td>
      <td>Nail Salon</td>
      <td>Chinese Restaurant</td>
      <td>Camera Store</td>
      <td>Malay Restaurant</td>
      <td>Restaurant</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Kerrisdale</td>
      <td>Coffee Shop</td>
      <td>Chinese Restaurant</td>
      <td>Sandwich Place</td>
      <td>Pharmacy</td>
      <td>Bubble Tea Shop</td>
      <td>Bank</td>
      <td>Tea Room</td>
      <td>Sushi Restaurant</td>
      <td>Dessert Shop</td>
      <td>Pizza Place</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Kitsilano</td>
      <td>Bakery</td>
      <td>Japanese Restaurant</td>
      <td>Tea Room</td>
      <td>Coffee Shop</td>
      <td>Food Truck</td>
      <td>French Restaurant</td>
      <td>Ice Cream Shop</td>
      <td>Sushi Restaurant</td>
      <td>American Restaurant</td>
      <td>Thai Restaurant</td>
    </tr>
  </tbody>
</table>
</div>



### Cluster Neighbourhoods


```python
# set number of clusters
kclusters = 5

vnc_grouped_clustering = vnc_ws_grouped.drop('Neighbourhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(vnc_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 
```




    array([1, 4, 0, 0, 0, 0, 3, 2, 0, 0], dtype=int32)




```python
# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

vancouver_merged = ws_neig_geo

# merge toronto_grouped with Vancouver data to add latitude/longitude for each neighborhood
vancouver_merged = vancouver_merged.join(neighborhoods_venues_sorted.set_index('Neighbourhood'), on='Neighbourhood')

vancouver_merged.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighbourhood</th>
      <th>Borough</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Shaughnessy</td>
      <td>West Side</td>
      <td>49.251863</td>
      <td>-123.138023</td>
      <td>2</td>
      <td>French Restaurant</td>
      <td>Park</td>
      <td>Italian Restaurant</td>
      <td>Deli / Bodega</td>
      <td>Dessert Shop</td>
      <td>Dim Sum Restaurant</td>
      <td>Diner</td>
      <td>Falafel Restaurant</td>
      <td>Fast Food Restaurant</td>
      <td>Food &amp; Drink Shop</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Fairview</td>
      <td>West Side</td>
      <td>49.264113</td>
      <td>-123.126835</td>
      <td>0</td>
      <td>Coffee Shop</td>
      <td>Park</td>
      <td>Asian Restaurant</td>
      <td>Korean Restaurant</td>
      <td>Pharmacy</td>
      <td>Nail Salon</td>
      <td>Chinese Restaurant</td>
      <td>Camera Store</td>
      <td>Malay Restaurant</td>
      <td>Restaurant</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Oakridge</td>
      <td>West Side</td>
      <td>49.230829</td>
      <td>-123.131134</td>
      <td>3</td>
      <td>Israeli Restaurant</td>
      <td>Vietnamese Restaurant</td>
      <td>Pharmacy</td>
      <td>Café</td>
      <td>Sandwich Place</td>
      <td>Sushi Restaurant</td>
      <td>Convenience Store</td>
      <td>Fast Food Restaurant</td>
      <td>Food &amp; Drink Shop</td>
      <td>Deli / Bodega</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Marpole</td>
      <td>West Side</td>
      <td>49.209223</td>
      <td>-123.136150</td>
      <td>0</td>
      <td>Sushi Restaurant</td>
      <td>Chinese Restaurant</td>
      <td>Pizza Place</td>
      <td>Sandwich Place</td>
      <td>Japanese Restaurant</td>
      <td>Shanghai Restaurant</td>
      <td>Gas Station</td>
      <td>Falafel Restaurant</td>
      <td>Liquor Store</td>
      <td>Dim Sum Restaurant</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Kitsilano</td>
      <td>West Side</td>
      <td>49.269410</td>
      <td>-123.155267</td>
      <td>0</td>
      <td>Bakery</td>
      <td>Japanese Restaurant</td>
      <td>Tea Room</td>
      <td>Coffee Shop</td>
      <td>Food Truck</td>
      <td>French Restaurant</td>
      <td>Ice Cream Shop</td>
      <td>Sushi Restaurant</td>
      <td>American Restaurant</td>
      <td>Thai Restaurant</td>
    </tr>
  </tbody>
</table>
</div>




```python
# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=12)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(vancouver_merged['Latitude'], vancouver_merged['Longitude'], vancouver_merged['Neighbourhood'], vancouver_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><span style="color:#565656">Make this Notebook Trusted to load map: File -> Trust Notebook</span><iframe src="about:blank" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" data-html=PCFET0NUWVBFIGh0bWw+CjxoZWFkPiAgICAKICAgIDxtZXRhIGh0dHAtZXF1aXY9ImNvbnRlbnQtdHlwZSIgY29udGVudD0idGV4dC9odG1sOyBjaGFyc2V0PVVURi04IiAvPgogICAgPHNjcmlwdD5MX1BSRUZFUl9DQU5WQVMgPSBmYWxzZTsgTF9OT19UT1VDSCA9IGZhbHNlOyBMX0RJU0FCTEVfM0QgPSBmYWxzZTs8L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmpzIj48L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2FqYXguZ29vZ2xlYXBpcy5jb20vYWpheC9saWJzL2pxdWVyeS8xLjExLjEvanF1ZXJ5Lm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvanMvYm9vdHN0cmFwLm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9jZG5qcy5jbG91ZGZsYXJlLmNvbS9hamF4L2xpYnMvTGVhZmxldC5hd2Vzb21lLW1hcmtlcnMvMi4wLjIvbGVhZmxldC5hd2Vzb21lLW1hcmtlcnMuanMiPjwvc2NyaXB0PgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9jc3MvYm9vdHN0cmFwLm1pbi5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvY3NzL2Jvb3RzdHJhcC10aGVtZS5taW4uY3NzIi8+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vbWF4Y2RuLmJvb3RzdHJhcGNkbi5jb20vZm9udC1hd2Vzb21lLzQuNi4zL2Nzcy9mb250LWF3ZXNvbWUubWluLmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2NkbmpzLmNsb3VkZmxhcmUuY29tL2FqYXgvbGlicy9MZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy8yLjAuMi9sZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9yYXdnaXQuY29tL3B5dGhvbi12aXN1YWxpemF0aW9uL2ZvbGl1bS9tYXN0ZXIvZm9saXVtL3RlbXBsYXRlcy9sZWFmbGV0LmF3ZXNvbWUucm90YXRlLmNzcyIvPgogICAgPHN0eWxlPmh0bWwsIGJvZHkge3dpZHRoOiAxMDAlO2hlaWdodDogMTAwJTttYXJnaW46IDA7cGFkZGluZzogMDt9PC9zdHlsZT4KICAgIDxzdHlsZT4jbWFwIHtwb3NpdGlvbjphYnNvbHV0ZTt0b3A6MDtib3R0b206MDtyaWdodDowO2xlZnQ6MDt9PC9zdHlsZT4KICAgIAogICAgICAgICAgICA8c3R5bGU+ICNtYXBfZTYyZmI4ZGVlYjQ1NGQzMjk4ZTQ3YTM2ODgxMzM1MWYgewogICAgICAgICAgICAgICAgcG9zaXRpb24gOiByZWxhdGl2ZTsKICAgICAgICAgICAgICAgIHdpZHRoIDogMTAwLjAlOwogICAgICAgICAgICAgICAgaGVpZ2h0OiAxMDAuMCU7CiAgICAgICAgICAgICAgICBsZWZ0OiAwLjAlOwogICAgICAgICAgICAgICAgdG9wOiAwLjAlOwogICAgICAgICAgICAgICAgfQogICAgICAgICAgICA8L3N0eWxlPgogICAgICAgIAo8L2hlYWQ+Cjxib2R5PiAgICAKICAgIAogICAgICAgICAgICA8ZGl2IGNsYXNzPSJmb2xpdW0tbWFwIiBpZD0ibWFwX2U2MmZiOGRlZWI0NTRkMzI5OGU0N2EzNjg4MTMzNTFmIiA+PC9kaXY+CiAgICAgICAgCjwvYm9keT4KPHNjcmlwdD4gICAgCiAgICAKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGJvdW5kcyA9IG51bGw7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgdmFyIG1hcF9lNjJmYjhkZWViNDU0ZDMyOThlNDdhMzY4ODEzMzUxZiA9IEwubWFwKAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJ21hcF9lNjJmYjhkZWViNDU0ZDMyOThlNDdhMzY4ODEzMzUxZicsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB7Y2VudGVyOiBbNDkuMjYwODcyNCwtMTIzLjExMzk1MjldLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgem9vbTogMTIsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBtYXhCb3VuZHM6IGJvdW5kcywKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGxheWVyczogW10sCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB3b3JsZENvcHlKdW1wOiBmYWxzZSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGNyczogTC5DUlMuRVBTRzM4NTcKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgfSk7CiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciB0aWxlX2xheWVyXzRkMDVkMmVhYzM4YjRlMjI4NzE5ZGYzODAzZjVhZWZmID0gTC50aWxlTGF5ZXIoCiAgICAgICAgICAgICAgICAnaHR0cHM6Ly97c30udGlsZS5vcGVuc3RyZWV0bWFwLm9yZy97en0ve3h9L3t5fS5wbmcnLAogICAgICAgICAgICAgICAgewogICJhdHRyaWJ1dGlvbiI6IG51bGwsCiAgImRldGVjdFJldGluYSI6IGZhbHNlLAogICJtYXhab29tIjogMTgsCiAgIm1pblpvb20iOiAxLAogICJub1dyYXAiOiBmYWxzZSwKICAic3ViZG9tYWlucyI6ICJhYmMiCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2U2MmZiOGRlZWI0NTRkMzI5OGU0N2EzNjg4MTMzNTFmKTsKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl84MDQ1NTdiNTU5OTg0MDg4YTNjZGRlNWI3ZjQxNTA5YiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQ5LjI1MTg2MjYsLTEyMy4xMzgwMjI2XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiMwMGI1ZWIiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMDBiNWViIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2U2MmZiOGRlZWI0NTRkMzI5OGU0N2EzNjg4MTMzNTFmKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzJlOTE0MzQ5ZTU3MjRkYzQ4YjcwMTc2MjZmMDY2YmFiID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzEwZjk3YTI5OTQxMjQyOTM4NDY1YjY5NjhmZDYwYzA5ID0gJCgnPGRpdiBpZD0iaHRtbF8xMGY5N2EyOTk0MTI0MjkzODQ2NWI2OTY4ZmQ2MGMwOSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U2hhdWdobmVzc3kgQ2x1c3RlciAyPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8yZTkxNDM0OWU1NzI0ZGM0OGI3MDE3NjI2ZjA2NmJhYi5zZXRDb250ZW50KGh0bWxfMTBmOTdhMjk5NDEyNDI5Mzg0NjViNjk2OGZkNjBjMDkpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfODA0NTU3YjU1OTk4NDA4OGEzY2RkZTViN2Y0MTUwOWIuYmluZFBvcHVwKHBvcHVwXzJlOTE0MzQ5ZTU3MjRkYzQ4YjcwMTc2MjZmMDY2YmFiKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzMyMDYzODNlMzRlMTQxNGQ4M2Q5M2JlMDkzNmRiNWU0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDkuMjY0MTEyOCwtMTIzLjEyNjgzNTJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZTYyZmI4ZGVlYjQ1NGQzMjk4ZTQ3YTM2ODgxMzM1MWYpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNDBlYzdlMTZlODM3NDI2Nzg4NWVhMjhlNTdmYWI3YWYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMTI0YTRlMDY4MjhiNDU1NjgzODk0M2VkODA4N2ZkYjEgPSAkKCc8ZGl2IGlkPSJodG1sXzEyNGE0ZTA2ODI4YjQ1NTY4Mzg5NDNlZDgwODdmZGIxIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5GYWlydmlldyBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzQwZWM3ZTE2ZTgzNzQyNjc4ODVlYTI4ZTU3ZmFiN2FmLnNldENvbnRlbnQoaHRtbF8xMjRhNGUwNjgyOGI0NTU2ODM4OTQzZWQ4MDg3ZmRiMSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8zMjA2MzgzZTM0ZTE0MTRkODNkOTNiZTA5MzZkYjVlNC5iaW5kUG9wdXAocG9wdXBfNDBlYzdlMTZlODM3NDI2Nzg4NWVhMjhlNTdmYWI3YWYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfN2FlOWM5MTZiYTZiNDA1Yjk0ODRkNDRkMmQxYjBhZmMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0OS4yMzA4Mjg4LC0xMjMuMTMxMTM0Ml0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODBmZmI0IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwZmZiNCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9lNjJmYjhkZWViNDU0ZDMyOThlNDdhMzY4ODEzMzUxZik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kM2I5ZmRhNDcyN2M0Mjc5YThmNGNkYjk5MmYxOGFlNyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85M2FlZDM3M2RlZTc0NTZjOWY2ZjMwZDk0OWQ4NjE3MiA9ICQoJzxkaXYgaWQ9Imh0bWxfOTNhZWQzNzNkZWU3NDU2YzlmNmYzMGQ5NDlkODYxNzIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk9ha3JpZGdlIENsdXN0ZXIgMzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZDNiOWZkYTQ3MjdjNDI3OWE4ZjRjZGI5OTJmMThhZTcuc2V0Q29udGVudChodG1sXzkzYWVkMzczZGVlNzQ1NmM5ZjZmMzBkOTQ5ZDg2MTcyKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzdhZTljOTE2YmE2YjQwNWI5NDg0ZDQ0ZDJkMWIwYWZjLmJpbmRQb3B1cChwb3B1cF9kM2I5ZmRhNDcyN2M0Mjc5YThmNGNkYjk5MmYxOGFlNyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9lMmNmOTQ2M2EwMTU0Mjk4OWRmZmFhNDI0ODRlY2Y1ZiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQ5LjIwOTIyMzMsLTEyMy4xMzYxNDk1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2U2MmZiOGRlZWI0NTRkMzI5OGU0N2EzNjg4MTMzNTFmKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2EwMTE2ZjQ5MDM0NTRjOTE4MTBhZmJmNTJiNWFjZDFlID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzQ2YWZlNWExNjNmMTQ5ODE4M2E4N2MzM2YwY2ExYjlmID0gJCgnPGRpdiBpZD0iaHRtbF80NmFmZTVhMTYzZjE0OTgxODNhODdjMzNmMGNhMWI5ZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TWFycG9sZSBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2EwMTE2ZjQ5MDM0NTRjOTE4MTBhZmJmNTJiNWFjZDFlLnNldENvbnRlbnQoaHRtbF80NmFmZTVhMTYzZjE0OTgxODNhODdjMzNmMGNhMWI5Zik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lMmNmOTQ2M2EwMTU0Mjk4OWRmZmFhNDI0ODRlY2Y1Zi5iaW5kUG9wdXAocG9wdXBfYTAxMTZmNDkwMzQ1NGM5MTgxMGFmYmY1MmI1YWNkMWUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfN2ZlMzE4ZDUwZjFhNDhkYzg2MDJkZGNlMmE2MGU5MWEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0OS4yNjk0MDk5LC0xMjMuMTU1MjY3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2U2MmZiOGRlZWI0NTRkMzI5OGU0N2EzNjg4MTMzNTFmKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzI3MzdlNDRjNWU5MDRiY2JhNDRjYWJkZDk5MjlmNzljID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2M2YWMyMWYxNzJiYzQ5ZDg5MmU0ZmM4YTliMGJmNjg3ID0gJCgnPGRpdiBpZD0iaHRtbF9jNmFjMjFmMTcyYmM0OWQ4OTJlNGZjOGE5YjBiZjY4NyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+S2l0c2lsYW5vIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMjczN2U0NGM1ZTkwNGJjYmE0NGNhYmRkOTkyOWY3OWMuc2V0Q29udGVudChodG1sX2M2YWMyMWYxNzJiYzQ5ZDg5MmU0ZmM4YTliMGJmNjg3KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzdmZTMxOGQ1MGYxYTQ4ZGM4NjAyZGRjZTJhNjBlOTFhLmJpbmRQb3B1cChwb3B1cF8yNzM3ZTQ0YzVlOTA0YmNiYTQ0Y2FiZGQ5OTI5Zjc5Yyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9iMDZjMmIyYWE5YjU0NGNiYWFiMDljNTgyMzJiZWZiMyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQ5LjIzNDY3MjgsLTEyMy4xNTUzODkzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2U2MmZiOGRlZWI0NTRkMzI5OGU0N2EzNjg4MTMzNTFmKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2IyMDM5N2YxNjU0ZTQ3YWFhY2E0NGFhN2QyN2NmMzVjID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzRmMjMyYWMwNmY1MDRlODZhOWQ1YjM3YTIzZmYzN2U4ID0gJCgnPGRpdiBpZD0iaHRtbF80ZjIzMmFjMDZmNTA0ZTg2YTlkNWIzN2EyM2ZmMzdlOCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+S2VycmlzZGFsZSBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2IyMDM5N2YxNjU0ZTQ3YWFhY2E0NGFhN2QyN2NmMzVjLnNldENvbnRlbnQoaHRtbF80ZjIzMmFjMDZmNTA0ZTg2YTlkNWIzN2EyM2ZmMzdlOCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9iMDZjMmIyYWE5YjU0NGNiYWFiMDljNTgyMzJiZWZiMy5iaW5kUG9wdXAocG9wdXBfYjIwMzk3ZjE2NTRlNDdhYWFjYTQ0YWE3ZDI3Y2YzNWMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNzU1MmMyNGQzY2E5NDFmN2E2ZmM5ZjlmNjdlNzcwMTEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0OS4yNjQ0ODQzLC0xMjMuMTg1NDMyNl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9lNjJmYjhkZWViNDU0ZDMyOThlNDdhMzY4ODEzMzUxZik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF80NzEzMzEzMTFiMzk0ZTIyYjdmMDU3ZGZiZjkxYzczYiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85MTIyMzU5NDJlZDU0ZTQ0YWM4NGM5NjIzMTA5OGY4NiA9ICQoJzxkaXYgaWQ9Imh0bWxfOTEyMjM1OTQyZWQ1NGU0NGFjODRjOTYyMzEwOThmODYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPldlc3QgUG9pbnQgR3JleSBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzQ3MTMzMTMxMWIzOTRlMjJiN2YwNTdkZmJmOTFjNzNiLnNldENvbnRlbnQoaHRtbF85MTIyMzU5NDJlZDU0ZTQ0YWM4NGM5NjIzMTA5OGY4Nik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl83NTUyYzI0ZDNjYTk0MWY3YTZmYzlmOWY2N2U3NzAxMS5iaW5kUG9wdXAocG9wdXBfNDcxMzMxMzExYjM5NGUyMmI3ZjA1N2RmYmY5MWM3M2IpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOTE4MjZhNjlmNjA3NDg4NGE1Njg1Mjg1ZjYyYjlkMjEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0OS4yNDA5Njc3LC0xMjMuMTY3MDAwOF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODAwMGZmIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9lNjJmYjhkZWViNDU0ZDMyOThlNDdhMzY4ODEzMzUxZik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8xYmUxY2E4YjNlNjc0YmE4YTY1YjIxZjg2YWE5NzQxYiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9iNjI1OGNkODFhNmY0ZDk5OTkxMmNhNGU0NWI2ZmZiZSA9ICQoJzxkaXYgaWQ9Imh0bWxfYjYyNThjZDgxYTZmNGQ5OTk5MTJjYTRlNDViNmZmYmUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkFyYnV0dXMgUmlkZ2UgQ2x1c3RlciAxPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8xYmUxY2E4YjNlNjc0YmE4YTY1YjIxZjg2YWE5NzQxYi5zZXRDb250ZW50KGh0bWxfYjYyNThjZDgxYTZmNGQ5OTk5MTJjYTRlNDViNmZmYmUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOTE4MjZhNjlmNjA3NDg4NGE1Njg1Mjg1ZjYyYjlkMjEuYmluZFBvcHVwKHBvcHVwXzFiZTFjYThiM2U2NzRiYThhNjViMjFmODZhYTk3NDFiKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzI0MzVmNDZjNzU1NzQ5Mjk5ZTBmNTczZjVlNDBiYTNmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDkuMjQ2Njg0NywtMTIzLjEyMDkxNV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9lNjJmYjhkZWViNDU0ZDMyOThlNDdhMzY4ODEzMzUxZik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9mYjE1YzNmYzI2YjQ0ODhmYTNmNThlOTgwNzZiYTg1ZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80MzhmMzAwZDYzMmU0NGUzOWNiMDU5Mzg3NTI0ZjZlOSA9ICQoJzxkaXYgaWQ9Imh0bWxfNDM4ZjMwMGQ2MzJlNDRlMzljYjA1OTM4NzUyNGY2ZTkiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlNvdXRoIENhbWJpZSBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2ZiMTVjM2ZjMjZiNDQ4OGZhM2Y1OGU5ODA3NmJhODVmLnNldENvbnRlbnQoaHRtbF80MzhmMzAwZDYzMmU0NGUzOWNiMDU5Mzg3NTI0ZjZlOSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8yNDM1ZjQ2Yzc1NTc0OTI5OWUwZjU3M2Y1ZTQwYmEzZi5iaW5kUG9wdXAocG9wdXBfZmIxNWMzZmMyNmI0NDg4ZmEzZjU4ZTk4MDc2YmE4NWYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZGVmODY5ZjdjN2JjNDhhZTg4NDlhMDJlM2ZiNDkxZjYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0OS4yNTM0NjAxLC0xMjMuMTg1MDQzOV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmZiMzYwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmYjM2MCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9lNjJmYjhkZWViNDU0ZDMyOThlNDdhMzY4ODEzMzUxZik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kNjFhYmVhMTU0ZWI0ZjhhYTBjYzVjODMwOWY0Yjc1YiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9jM2FhN2RjZTA0NzM0YzEzOWMxYjQ4NWY3MTc4YTM2NSA9ICQoJzxkaXYgaWQ9Imh0bWxfYzNhYTdkY2UwNDczNGMxMzljMWI0ODVmNzE3OGEzNjUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkR1bmJhci1Tb3V0aGxhbmRzIENsdXN0ZXIgNDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZDYxYWJlYTE1NGViNGY4YWEwY2M1YzgzMDlmNGI3NWIuc2V0Q29udGVudChodG1sX2MzYWE3ZGNlMDQ3MzRjMTM5YzFiNDg1ZjcxNzhhMzY1KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2RlZjg2OWY3YzdiYzQ4YWU4ODQ5YTAyZTNmYjQ5MWY2LmJpbmRQb3B1cChwb3B1cF9kNjFhYmVhMTU0ZWI0ZjhhYTBjYzVjODMwOWY0Yjc1Yik7CgogICAgICAgICAgICAKICAgICAgICAKPC9zY3JpcHQ+ onload="this.contentDocument.open();this.contentDocument.write(atob(this.getAttribute('data-html')));this.contentDocument.close();" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



### Analysis<a name="analysis"></a>

#### Examining the resulting Clusters

#### Cluster 1


```python
vancouver_merged.loc[vancouver_merged['Cluster Labels'] == 0, vancouver_merged.columns[[1] + list(range(5, vancouver_merged.shape[1]))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>West Side</td>
      <td>Coffee Shop</td>
      <td>Park</td>
      <td>Asian Restaurant</td>
      <td>Korean Restaurant</td>
      <td>Pharmacy</td>
      <td>Nail Salon</td>
      <td>Chinese Restaurant</td>
      <td>Camera Store</td>
      <td>Malay Restaurant</td>
      <td>Restaurant</td>
    </tr>
    <tr>
      <th>3</th>
      <td>West Side</td>
      <td>Sushi Restaurant</td>
      <td>Chinese Restaurant</td>
      <td>Pizza Place</td>
      <td>Sandwich Place</td>
      <td>Japanese Restaurant</td>
      <td>Shanghai Restaurant</td>
      <td>Gas Station</td>
      <td>Falafel Restaurant</td>
      <td>Liquor Store</td>
      <td>Dim Sum Restaurant</td>
    </tr>
    <tr>
      <th>4</th>
      <td>West Side</td>
      <td>Bakery</td>
      <td>Japanese Restaurant</td>
      <td>Tea Room</td>
      <td>Coffee Shop</td>
      <td>Food Truck</td>
      <td>French Restaurant</td>
      <td>Ice Cream Shop</td>
      <td>Sushi Restaurant</td>
      <td>American Restaurant</td>
      <td>Thai Restaurant</td>
    </tr>
    <tr>
      <th>5</th>
      <td>West Side</td>
      <td>Coffee Shop</td>
      <td>Chinese Restaurant</td>
      <td>Sandwich Place</td>
      <td>Pharmacy</td>
      <td>Bubble Tea Shop</td>
      <td>Bank</td>
      <td>Tea Room</td>
      <td>Sushi Restaurant</td>
      <td>Dessert Shop</td>
      <td>Pizza Place</td>
    </tr>
    <tr>
      <th>6</th>
      <td>West Side</td>
      <td>Coffee Shop</td>
      <td>Japanese Restaurant</td>
      <td>Café</td>
      <td>Sushi Restaurant</td>
      <td>Pub</td>
      <td>Vegetarian / Vegan Restaurant</td>
      <td>Bookstore</td>
      <td>Pizza Place</td>
      <td>Fast Food Restaurant</td>
      <td>Liquor Store</td>
    </tr>
    <tr>
      <th>8</th>
      <td>West Side</td>
      <td>Coffee Shop</td>
      <td>Light Rail Station</td>
      <td>Bus Stop</td>
      <td>Shopping Mall</td>
      <td>Malay Restaurant</td>
      <td>Cantonese Restaurant</td>
      <td>Sushi Restaurant</td>
      <td>Liquor Store</td>
      <td>Café</td>
      <td>Juice Bar</td>
    </tr>
  </tbody>
</table>
</div>



#### Cluster 2


```python
vancouver_merged.loc[vancouver_merged['Cluster Labels'] == 1, vancouver_merged.columns[[1] + list(range(5, vancouver_merged.shape[1]))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>West Side</td>
      <td>Spa</td>
      <td>Bakery</td>
      <td>Grocery Store</td>
      <td>Nightlife Spot</td>
      <td>Yoga Studio</td>
      <td>Deli / Bodega</td>
      <td>Dessert Shop</td>
      <td>Dim Sum Restaurant</td>
      <td>Diner</td>
      <td>Falafel Restaurant</td>
    </tr>
  </tbody>
</table>
</div>



#### Cluster 3


```python
vancouver_merged.loc[vancouver_merged['Cluster Labels'] == 2, vancouver_merged.columns[[1] + list(range(5, vancouver_merged.shape[1]))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>West Side</td>
      <td>French Restaurant</td>
      <td>Park</td>
      <td>Italian Restaurant</td>
      <td>Deli / Bodega</td>
      <td>Dessert Shop</td>
      <td>Dim Sum Restaurant</td>
      <td>Diner</td>
      <td>Falafel Restaurant</td>
      <td>Fast Food Restaurant</td>
      <td>Food &amp; Drink Shop</td>
    </tr>
  </tbody>
</table>
</div>



#### Cluster 4


```python
vancouver_merged.loc[vancouver_merged['Cluster Labels'] == 3, vancouver_merged.columns[[1] + list(range(5, vancouver_merged.shape[1]))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>West Side</td>
      <td>Israeli Restaurant</td>
      <td>Vietnamese Restaurant</td>
      <td>Pharmacy</td>
      <td>Café</td>
      <td>Sandwich Place</td>
      <td>Sushi Restaurant</td>
      <td>Convenience Store</td>
      <td>Fast Food Restaurant</td>
      <td>Food &amp; Drink Shop</td>
      <td>Deli / Bodega</td>
    </tr>
  </tbody>
</table>
</div>



#### Cluster 5


```python
vancouver_merged.loc[vancouver_merged['Cluster Labels'] == 4, vancouver_merged.columns[[1] + list(range(5, vancouver_merged.shape[1]))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9</th>
      <td>West Side</td>
      <td>Coffee Shop</td>
      <td>Ice Cream Shop</td>
      <td>Sushi Restaurant</td>
      <td>Indian Restaurant</td>
      <td>Italian Restaurant</td>
      <td>Gym Pool</td>
      <td>Gym / Fitness Center</td>
      <td>Cosmetics Shop</td>
      <td>Deli / Bodega</td>
      <td>Dessert Shop</td>
    </tr>
  </tbody>
</table>
</div>



## Results and Discussion <a name="results"></a>

The objective of the business problem was to help stakeholders identify one of the safest borough in Vancouver, and an appropriate neighborhood within the borough to set up a commercial establishment especially a Grocery store. This has been achieved by first making use of Vancouver crime data to identify a safe borugh with considerable number of neighborhood for any business to be viable. After selecting the borough it was imperative to choose the right neighborhood where grocery shops were not among venues in a close proximity to each other. We achieved this by grouping the neighborhoods into clusters to assist the stakeholders by providing them with relavent data about venues and safety of a given neighborhood.

## Conclusion <a name="conclusion"></a>

We have explored the crime data to understand different types of crimes in all neighborhoods of Vancouver and later categorized them into different boroughs, this helped us group the neighborhoods into boroughs and choose the safest borough first. Once we confirmed the borough the number of neighborhoods for consideration also comes down, we further shortlist the neighborhoods based on the common venues, to choose a neighborhood which best suits the business problem.


```python

```
