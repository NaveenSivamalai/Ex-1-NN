### ENTER YOUR NAME : NAVEEN S
### ENTER YOUR REGISTER NO : 212222110030
### EX. NO.1
### DATE : 25/2/2024
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv("Churn_Modelling.csv")
data
data.head()

X=data.iloc[:,:-1].values
X

y=data.iloc[:,-1].values
y

data.isnull().sum()

data.duplicated()

data.describe()

data = data.drop(['Surname', 'Geography','Gender'], axis=1)
data.head()

scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)

X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)

X_train

X_test

print("Lenght of X_test ",len(X_test))

```


## OUTPUT:
## DATASET

![307612272-fb93df05-0d85-46cd-8b72-f1b5dbb2f45e](https://github.com/NaveenSivamalai/Ex-1-NN/assets/123792574/9adce84b-4797-4d34-b18d-da063e6c155c)

## X VALUES

![307612320-6f82f7d9-a77a-4b07-b1c1-9e00adeff7ee](https://github.com/NaveenSivamalai/Ex-1-NN/assets/123792574/ace117f3-050d-4190-bd13-78d6f07e5065)

## Y VALUES

![307612361-3401a399-4672-416c-9735-0762b076671e](https://github.com/NaveenSivamalai/Ex-1-NN/assets/123792574/1dd9bbfe-95cc-4df7-bff1-d283aded1ec0)

## NULL
![307612398-d91cc7c2-d007-4015-b247-fb31ac81acf4](https://github.com/NaveenSivamalai/Ex-1-NN/assets/123792574/2a09a528-a8d9-4bae-8ad1-b2e252d82361)


## DUPLICATE
![307612409-a4ee7771-e2c4-4c37-89b3-975b225f734c](https://github.com/NaveenSivamalai/Ex-1-NN/assets/123792574/b8d52611-ebfe-4e64-9189-0dc550554a71)


## DESCRIBE

![307612438-27bb11f8-6d9f-4f46-9ee4-4b3a4ab026d1](https://github.com/NaveenSivamalai/Ex-1-NN/assets/123792574/47fe1718-301d-4dd7-872b-f68d03de853b)

## DATASET AFTER DROPPING

![307612479-f224cb6d-38d7-43a1-9fbf-41f927583e23](https://github.com/NaveenSivamalai/Ex-1-NN/assets/123792574/4d7662ce-9006-467f-a517-c7e549381813)

## NORMALIZE DATASET

![307612536-ed14c22a-b340-481d-8f96-240e0e18c8e7](https://github.com/NaveenSivamalai/Ex-1-NN/assets/123792574/465a9a04-8f8f-4c50-b750-35b5649f0399)

## X TRAIN

![307612579-69fa3229-5788-4c90-86e8-9614d6b00aaa](https://github.com/NaveenSivamalai/Ex-1-NN/assets/123792574/ea257fde-01c3-45dc-bec5-a894859719e8)

## X TEST
![307612595-6dbf4cca-969f-419d-82d4-f95c91a9e18d](https://github.com/NaveenSivamalai/Ex-1-NN/assets/123792574/71f1e60d-94d0-4a6e-b105-7b01e3531c08)


## LENGTH
![307612602-63aa77f8-f224-4657-8f1c-a79496aa59bc](https://github.com/NaveenSivamalai/Ex-1-NN/assets/123792574/fea78484-58f6-4bfa-a511-e5989fef1689)




## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


