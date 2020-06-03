```python
%load_ext sql
```

    The sql extension is already loaded. To reload it, use:
      %reload_ext sql



```python

%sql ibm_db_sa://jjh41429:c08%40d822wd8cds9d@dashdb-txn-sbox-yp-dal09-08.services.dal.bluemix.net:50000/BLUDB
```




    'Connected: jjh41429@BLUDB'



Problem 1

Find the total number of crimes recorded in the crime table.


```python
%sql select count(*) as num_crimes from CHICAGO_CRIME_DATA

```

     * ibm_db_sa://jjh41429:***@dashdb-txn-sbox-yp-dal09-08.services.dal.bluemix.net:50000/BLUDB
    Done.





<table>
    <tr>
        <th>num_crimes</th>
    </tr>
    <tr>
        <td>0</td>
    </tr>
</table>



Problem 2: 

Retrieve first 10 rows from the CRIME table.




```python
%sql select * from CHICAGO_CRIME_DATA limit 10

```

     * ibm_db_sa://jjh41429:***@dashdb-txn-sbox-yp-dal09-08.services.dal.bluemix.net:50000/BLUDB
    Done.





<table>
    <tr>
        <th>id</th>
        <th>case_number</th>
        <th>DATE</th>
        <th>block</th>
        <th>iucr</th>
        <th>primary_type</th>
        <th>description</th>
        <th>location_description</th>
        <th>arrest</th>
        <th>domestic</th>
        <th>beat</th>
        <th>district</th>
        <th>ward</th>
        <th>community_area_number</th>
        <th>fbicode</th>
        <th>x_coordinate</th>
        <th>y_coordinate</th>
        <th>YEAR</th>
        <th>updatedon</th>
        <th>latitude</th>
        <th>longitude</th>
        <th>location</th>
    </tr>
</table>




```python
import pandas as pd

```


```python
from bokeh.plotting import figure, output_file, show,output_notebook
output_notebook()
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    <ipython-input-3-0d6387fc4308> in <module>
    ----> 1 from bokeh.plotting import figure, output_file, show,output_notebook
          2 output_notebook()


    ModuleNotFoundError: No module named 'bokeh'



```python
def make_dashboard(x, gdp_change, unemployment, title, file_name):
    output_file(file_name)                                                                              #name of the file
    p = figure(title=title, x_axis_label='year', y_axis_label='%')                                      #plotting the dashboard
    p.line(x.squeeze(), gdp_change.squeeze(), color="firebrick", line_width=4, legend="% GDP change")   #plotting the 'GDP' part
    p.line(x.squeeze(), unemployment.squeeze(), color="green", line_width=4, legend="% unemployed")     #plotting the 'unemployment' part
    show(p)
```


```python
links={'GDP':'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/PY0101EN/projects/coursera_project/clean_gdp.csv',\
       'unemployment':'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/PY0101EN/projects/coursera_project/clean_unemployment.csv'}
```


```python
csv_path=links["GDP"]
d1=pd.read_csv(csv_path)            #defining the dataframe
d1.head()
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
      <th>date</th>
      <th>level-current</th>
      <th>level-chained</th>
      <th>change-current</th>
      <th>change-chained</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1948</td>
      <td>274.8</td>
      <td>2020.0</td>
      <td>-0.7</td>
      <td>-0.6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1949</td>
      <td>272.8</td>
      <td>2008.9</td>
      <td>10.0</td>
      <td>8.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1950</td>
      <td>300.2</td>
      <td>2184.0</td>
      <td>15.7</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1951</td>
      <td>347.3</td>
      <td>2360.0</td>
      <td>5.9</td>
      <td>4.1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1952</td>
      <td>367.7</td>
      <td>2456.1</td>
      <td>6.0</td>
      <td>4.7</td>
    </tr>
  </tbody>
</table>
</div>




```python
csv_path1=links["unemployment"]
d2=pd.read_csv(csv_path1)                   #defining the dataframe
d2.head() 
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
      <th>date</th>
      <th>unemployment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1948</td>
      <td>3.750000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1949</td>
      <td>6.050000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1950</td>
      <td>5.208333</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1951</td>
      <td>3.283333</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1952</td>
      <td>3.025000</td>
    </tr>
  </tbody>
</table>
</div>




```python
d3=d2[d2['unemployment']>8.5]                #extracting the part of the dataframe d2 to a new dataframe
d3
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
      <th>date</th>
      <th>unemployment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>34</th>
      <td>1982</td>
      <td>9.708333</td>
    </tr>
    <tr>
      <th>35</th>
      <td>1983</td>
      <td>9.600000</td>
    </tr>
    <tr>
      <th>61</th>
      <td>2009</td>
      <td>9.283333</td>
    </tr>
    <tr>
      <th>62</th>
      <td>2010</td>
      <td>9.608333</td>
    </tr>
    <tr>
      <th>63</th>
      <td>2011</td>
      <td>8.933333</td>
    </tr>
  </tbody>
</table>
</div>




```python
gdp_dataframe1=pd.read_csv(csv_path1)
x = pd.DataFrame(gdp_dataframe1, columns=['date'])
x.head()
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
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1948</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1949</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1950</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1951</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1952</td>
    </tr>
  </tbody>
</table>
</div>




```python
csv_path2=links['GDP']
gdp_dataframe2=pd.read_csv(csv_path2)
gdp_change = pd.DataFrame(gdp_dataframe2, columns=['change-current'])
gdp_change.head()
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
      <th>change-current</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5.9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
csv_path3=links['unemployment']
unemploy_dataframe1= pd.read_csv(csv_path3)
unemployment = pd.DataFrame(unemploy_dataframe1, columns=['unemployment'])
unemployment.head()
```


```python
title = "Unemployment stats according to GDP"
file_name = "index.html"
make_dashboard(x=x, gdp_change=gdp_change, unemployment=unemployment, title=title, file_name=file_name)

```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-1-0f2b3c535626> in <module>
          1 title = "Unemployment stats according to GDP"
          2 file_name = "index.html"
    ----> 3 make_dashboard(x=x, gdp_change=gdp_change, unemployment=unemployment, title=title, file_name=file_name)
    

    NameError: name 'make_dashboard' is not defined



```python
import pandas as pd
```


```python
data={'Name':['kelvin','jonna','kriz'],'Age':[20,21,44]}

```


```python
df=pd.DataFrame(data)
```


```python
df
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
      <th>Name</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>kelvin</td>
      <td>20</td>
    </tr>
    <tr>
      <th>1</th>
      <td>jonna</td>
      <td>21</td>
    </tr>
    <tr>
      <th>2</th>
      <td>kriz</td>
      <td>44</td>
    </tr>
  </tbody>
</table>
</div>




```python
data=[['kelvin',20],['sf',20]]
df=pd.DataFrame(data,columns=['name','age'])
df
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
      <th>name</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>kelvin</td>
      <td>20</td>
    </tr>
    <tr>
      <th>1</th>
      <td>sf</td>
      <td>20</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[['age','name']]
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
      <th>age</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20</td>
      <td>kelvin</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20</td>
      <td>sf</td>
    </tr>
  </tbody>
</table>
</div>




```python
import numpy as np
num=np.array([[1,2,3],[4,5,6],[7,8,9]])
num
```




    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])




```python
num[0,2]
```




    3




```python
a=np.arange(10).reshape(5,-3)
a
```




    array([[0, 1],
           [2, 3],
           [4, 5],
           [6, 7],
           [8, 9]])




```python
a
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
import numpy as np

q=np.arange(0,np.pi,0.1)
```


```python
q
```




    array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. , 1.1, 1.2,
           1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2. , 2.1, 2.2, 2.3, 2.4, 2.5,
           2.6, 2.7, 2.8, 2.9, 3. , 3.1])




```python
np.pi
```




    3.141592653589793




```python

```
