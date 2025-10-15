#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# ### Importing Data Set

# In[5]:


data = pd.read_csv('Z:\\Sasindu\\Data set\\loan_data_set.csv')


# In[6]:


df = pd.DataFrame(data)


# In[7]:


df.head()


# #### Informations of the Data set

# In[8]:


df.info()


# In[9]:


df.shape


# In[10]:


df.isnull().sum()


#  ##### Discriptive Statistics

# In[11]:


df.describe()


# #### Value Counts of categorical Data

# In[12]:


features = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
for feature in features:
    counts = df[feature].value_counts()
    print(f"Value counts of {feature} \n{counts}\n")


# ### Handling missing values

# ###### Dependents column type should be numerical.There was values 3+ and null, 3+ replace with 3 and fill null values from 0 ,because mode is 0 ,and converted df['Dependents'] to int type

# In[13]:


mode = df['Dependents'].mode()[0]
df['Dependents']=df['Dependents'].replace('3+','3').fillna(mode)


# In[14]:


df['Dependents']=df['Dependents'].astype('int32')


# ### Exploratory Data Analysis for handling missing values

# In[15]:


category_col = ['Gender','Married','Education','Self_Employed','Property_Area','Loan_Status']

for column in category_col:
    plt.figure(figsize=(5,8))
    sns.countplot(df[column])
    plt.xlabel(f'{column}')
    plt.ylabel('Count')
    plt.title(f'Count plot of the {column}')
    plt.show()


# In[16]:


numeric_vals = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']

for col in numeric_vals:
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(df[col], kde=True)
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df[col])
    plt.tight_layout()
    plt.show()


# In[17]:


df.isnull().sum()


# ##### Categorycal data filled with mode

# In[18]:


category_col = ['Gender','Married','Education','Self_Employed','Property_Area']

for column in category_col:
    globals()[f'mode_{column}'] = df[column].mode()[0]
    df[column].fillna(globals()[f'mode_{column}'], inplace=True)
    print(f"Mode of {column} :",globals()[f'mode_{column}'])


# In[19]:


df.isnull().sum()


# In[20]:


df['Credit_History'].value_counts()


# In[21]:


df['Loan_Amount_Term'].value_counts()


# ##### Loan_Amount_Term and Credit_History also like categorical data because thay haven't distributed distribution. So, Mode values of the each column can be added for missing values.

# In[22]:


numeric_category_col = ['Credit_History','Loan_Amount_Term']

for column in numeric_category_col:
    globals()[f'mode_{column}'] = df[column].mode()[0]
    df[column].fillna(globals()[f'mode_{column}'], inplace=True)
    print(f"Mode of {column} :",globals()[f'mode_{column}'])


# In[23]:


df.isnull().sum()


# ##### LoanAmount has skewed distribution. Usualy,The Median is used to fill null values for this type senarios. So, Missing values of the Loan Amount was filled with median value.  

# In[24]:


median_Loan_Amount = df['LoanAmount'].median()
df['LoanAmount'].fillna(median_Loan_Amount,inplace=True)


# In[25]:


df.isnull().sum()


# ##### Now, All the missing values have filled. Then, Move on to outlier ditection step.

# ### Remove Outliers.

# In[26]:


ApplicantIncome_out = df['ApplicantIncome']<=50000
CoapplicantIncome_out =df['CoapplicantIncome']<=20000
LoanAmount_out =df['LoanAmount']<=500


# In[27]:


df_no_outliers = df.copy()

df_no_outliers = df[(ApplicantIncome_out) & (CoapplicantIncome_out) & (LoanAmount_out)]


# In[28]:


df_no_outliers.info()


# ##### Actually, in this step, having domain knowledge is essential. By studying boxplot graphs, many outliers can be detected. However, these outliers might be actual values that could affect the model. Therefore, we remove the rows that have high variance.  

# ##### After removing outliers, there are 605 rows, Before there was 614 row. so, there is 9 rows has removed as outliers. 

# In[29]:


df_no_outliers=df_no_outliers.drop(columns ='Loan_ID',axis=1)


# In[30]:


df_no_outliers.head()


# ##### Loan_ID colunm is not need further more. So, that column was removed.

# ### Feature Enginearing.

# ##### 'Gender','Married','Education','Self_Employed','Property_Area', and 'Loan_Status' are binary categorical columns and 'Property_Area' is N-ary categorycal variables. Here 'Loan_Status' is predicted variable. So that colunm can be encorded by using label encording method.
# ##### For other binary categorycal variable, One hot or Label encording can be applied becouse they are not ordinal variables. But as my thought, If I use one hot encording , the data frame end up with high number of dimentions. So,to enhance Encording with low diamentions, the suitable way is doing label encording.
# ##### To apply encording method for Property_Area variable, domain knowladge is needed. I think no critiria for selected for loan or not. I reffered following graphs to obtain this. one hot encording method can be applied.

# In[31]:


plt.figure(figsize=(12,8))
plt.subplot(1, 2, 1)
sns.countplot(data = df, hue= df["Property_Area"],x=df['Loan_Status'])
plt.subplot(1, 2, 2)
sns.countplot(data = df, x= df["Property_Area"],hue=df['Loan_Status'])
plt.tight_layout()
plt.show()


# #### Label encoding for binary categorical features

# In[32]:


from sklearn.preprocessing import LabelEncoder


# In[33]:


LE = LabelEncoder()
cate_cols_LE = ['Gender','Married','Education','Self_Employed','Loan_Status']
df_encoded = df_no_outliers.copy()
for col in cate_cols_LE:
    df_encoded[col] = LE.fit_transform(df_encoded[col])


# In[34]:


df_encoded.head()


# #### One hot Encording for Property_Area 

# In[35]:


df_encoded = pd.get_dummies(df_encoded, columns=['Property_Area'])


# In[36]:


df_encoded.head()


# ### Splitting Data

# In[37]:


x = df_encorded.drop(columns ='Loan_Status',axis =1).values
y = df_encorded['Loan_Status'].values


# In[38]:


from sklearn.model_selection import train_test_split


# In[39]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)


# ##### 20% of the whole data set was used as test set 

# ### Scaling Data

# ##### Typicaly, If the Data set is Normaly distributed , Satanderd scaler is used. If not,Minmax scaler is used.
# ##### In this senario, categorical columns not need scaling , but numerical columns has skewed distribution, normaly minmax scaling used to skewed distributions.

# In[40]:


from sklearn.preprocessing import MinMaxScaler


# In[41]:


mm = MinMaxScaler()
x_train_scaled = mm.fit_transform(x_train)
x_test_scaled = mm.transform(x_test)


# ### PCA 

# ##### In this step, Enhancing performance and dimentionality reduction are expected. Selecting number of components in PCA is cruitiol. For that, scree plot can be used. In a scree plot is selected based on where the plot shows an elbow or inflection point

# In[42]:


from sklearn.decomposition import PCA


# In[43]:


pca = PCA()
pca.fit(x_train_scaled)

explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = explained_variance_ratio.cumsum()

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_explained_variance, marker='o', linestyle='--')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs. Number of Components')
plt.show()


# #### Apply PCA

# In[44]:


pca = PCA(n_components=8)
x_train_pca = pca.fit_transform(x_train_scaled)
x_test_pca = pca.transform(x_test_scaled)


# ##### By studing above graph , 8 is the best number of components.

# ### Define Model

# ##### As a superviced learning classification problem, logistic regression was applied.

# In[45]:


def evaluate_model(model):
    """
    Train the model on training data and print its accuracy on test data.
    """
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    print(f"{model} --> {acc}")


# #### Logistic Regression

# In[46]:


from sklearn.linear_model import LogisticRegression
lo = LogisticRegression()
model_acc(lo)


# #### Decision Tree 

# In[47]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
model_acc(dt)


# #### Random Forest

# In[48]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
model_acc(rf)


# ### Train Model, Predict, and Calculating Model Accuracy

# ###### Here, PCA transformed data set was trained using Logistic regression, Decision tree, and Random forest. Then, Test set was used to predict y values and calculate accuracy for each models. from that best model can be choosen. Here we have to study accuracy , confution matrix   and classification report.

# In[49]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
models = [
    ('Logistic Regression', LogisticRegression()),
    ('Decision Tree', DecisionTreeClassifier()),
    ('Random Forest', RandomForestClassifier())
]

for model_name, model in models:
    
    model.fit(x_train_pca,y_train)
    
    y_train_pred = model.predict(x_train_pca)
    y_test_pred = model.predict(x_test_pca)
    
    train_accuracy = accuracy_score(y_train,y_train_pred)
    test_accuracy = accuracy_score(y_test,y_test_pred)
    conf_matrix = confusion_matrix(y_test,y_test_pred)
    class_report = classification_report(y_test, y_test_pred)
    
    print(model_name)
    print('Train_accuracy :',train_accuracy)
    print('Test_accuracy :\n',test_accuracy)
    print('Confusion_matrix :\n',conf_matrix)
    print('Classification_report :\n',class_report)


# ### Train Model, Predict, and Calculating Model Accuracy for scaled data (without PCA transformed)

# In[50]:


for model_name, model in models:
    
    model.fit(x_train_scaled,y_train)
    
    y_train_pred = model.predict(x_train_scaled)
    y_test_pred = model.predict(x_test_scaled)
    
    train_accuracy = accuracy_score(y_train,y_train_pred)
    test_accuracy = accuracy_score(y_test,y_test_pred)
    conf_matrix = confusion_matrix(y_test,y_test_pred)
    class_report = classification_report(y_test, y_test_pred)
    
    print(model_name)
    print('Train_accuracy :',train_accuracy)
    print('Test_accuracy :\n',test_accuracy)
    print('Confusion_matrix :\n',conf_matrix)
    print('Classification_report :\n',class_report)


# #### Train Model, Predict, and Calculating Model Accuracy (Without scaled & PCA transformed)

# In[51]:


for model_name, model in models:
    
    model.fit(x_train,y_train)
    
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    
    train_accuracy = accuracy_score(y_train,y_train_pred)
    test_accuracy = accuracy_score(y_test,y_test_pred)
    conf_matrix = confusion_matrix(y_test,y_test_pred)
    class_report = classification_report(y_test, y_test_pred)
    
    print(model_name)
    print('Train_accuracy :',train_accuracy)
    print('Test_accuracy :\n',test_accuracy)
    print('Confusion_matrix :\n',conf_matrix)
    print('Classification_report :\n',class_report)


# ###### By studing accuracy factors, best model was given by logistic regression with scaled and PCA transformed data. It's accuracy is about 85.95% and precision is also high.

df['Loan_Amount_Term'].value_counts()


# ##### Loan_Amount_Term and Credit_History also like categorical data because thay haven't distributed distribution. So, Mode values of the each column can be added for missing values.

# In[22]:


numeric_category_col = ['Credit_History','Loan_Amount_Term']

for column in numeric_category_col:
    globals()[f'mode_{column}'] = df[column].mode()[0]
    df[column].fillna(globals()[f'mode_{column}'], inplace=True)
    print(f"Mode of {column} :",globals()[f'mode_{column}'])


# In[23]:


df.isnull().sum()


# ##### LoanAmount has skewed distribution. Usualy,The Median is used to fill null values for this type senarios. So, Missing values of the Loan Amount was filled with median value.  

# In[24]:


median_Loan_Amount = df['LoanAmount'].median()
df['LoanAmount'].fillna(median_Loan_Amount,inplace=True)


# In[25]:


df.isnull().sum()


# ##### Now, All the missing values have filled. Then, Move on to outlier ditection step.

# ### Remove Outliers.

# In[26]:


ApplicantIncome_out = df['ApplicantIncome']<=50000
CoapplicantIncome_out =df['CoapplicantIncome']<=20000
LoanAmount_out =df['LoanAmount']<=500


# In[27]:


df_no_outliers = df.copy()

df_no_outliers = df[(ApplicantIncome_out) & (CoapplicantIncome_out) & (LoanAmount_out)]


# In[28]:


df_no_outliers.info()


# ##### Actually, in this step, having domain knowledge is essential. By studying boxplot graphs, many outliers can be detected. However, these outliers might be actual values that could affect the model. Therefore, we remove the rows that have high variance.  

# ##### After removing outliers, there are 605 rows, Before there was 614 row. so, there is 9 rows has removed as outliers. 

# In[29]:


df_no_outliers=df_no_outliers.drop(columns ='Loan_ID',axis=1)


# In[30]:


df_no_outliers.head()


# ##### Loan_ID colunm is not need further more. So, that column was removed.

# ### Feature Enginearing.

# ##### 'Gender','Married','Education','Self_Employed','Property_Area', and 'Loan_Status' are binary categorical columns and 'Property_Area' is N-ary categorycal variables. Here 'Loan_Status' is predicted variable. So that colunm can be encorded by using label encording method.
# ##### For other binary categorycal variable, One hot or Label encording can be applied becouse they are not ordinal variables. But as my thought, If I use one hot encording , the data frame end up with high number of dimentions. So,to enhance Encording with low diamentions, the suitable way is doing label encording.
# ##### To apply encording method for Property_Area variable, domain knowladge is needed. I think no critiria for selected for loan or not. I reffered following graphs to obtain this. one hot encording method can be applied.

# In[31]:


plt.figure(figsize=(12,8))
plt.subplot(1, 2, 1)
sns.countplot(data = df, hue= df["Property_Area"],x=df['Loan_Status'])
plt.subplot(1, 2, 2)
sns.countplot(data = df, x= df["Property_Area"],hue=df['Loan_Status'])
plt.tight_layout()
plt.show()


# #### Lable encording to binary categorical features

# In[32]:


from sklearn.preprocessing import LabelEncoder


# In[33]:


LE = LabelEncoder()
cate_cols_LE = ['Gender','Married','Education','Self_Employed','Loan_Status']
df_encorded = df_no_outliers.copy()
for col in cate_cols_LE:
    df_encorded[col] = LE.fit_transform(df_encorded[col])


# In[34]:


df_encorded.head()


# #### One hot Encording for Property_Area 

# In[35]:


df_encorded = pd.get_dummies(df_encorded, columns=['Property_Area'])


# In[36]:


df_encorded.head()


# ### Splitting Data

# In[37]:


x = df_encorded.drop(columns ='Loan_Status',axis =1).values
y = df_encorded['Loan_Status'].values


# In[38]:


from sklearn.model_selection import train_test_split


# In[39]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)


# ##### 20% of the whole data set was used as test set 

# ### Scaling Data

# ##### Typicaly, If the Data set is Normaly distributed , Satanderd scaler is used. If not,Minmax scaler is used.
# ##### In this senario, categorical columns not need scaling , but numerical columns has skewed distribution, normaly minmax scaling used to skewed distributions.

# In[40]:


from sklearn.preprocessing import MinMaxScaler


# In[41]:


mm = MinMaxScaler()
x_train_scaled = mm.fit_transform(x_train)
x_test_scaled = mm.transform(x_test)


# ### PCA 

# ##### In this step, Enhancing performance and dimentionality reduction are expected. Selecting number of components in PCA is cruitiol. For that, scree plot can be used. In a scree plot is selected based on where the plot shows an elbow or inflection point

# In[42]:


from sklearn.decomposition import PCA


# In[43]:


pca = PCA()
pca.fit(x_train_scaled)

explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = explained_variance_ratio.cumsum()

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_explained_variance, marker='o', linestyle='--')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs. Number of Components')
plt.show()


# #### Apply PCA

# In[44]:


pca = PCA(n_components=8)
x_train_pca = pca.fit_transform(x_train_scaled)
x_test_pca = pca.transform(x_test_scaled)


# ##### By studing above graph , 8 is the best number of components.

# ### Define Model

# ##### As a superviced learning classification problem, logistic regression was applied.

# In[45]:


def model_acc(model):
    model.fit(x_train,y_train)
    acc = model.score(x_test, y_test)
    print(str(model)+'-->'+str(acc))


# #### Logistic Regression

# In[46]:


from sklearn.linear_model import LogisticRegression
lo = LogisticRegression()
model_acc(lo)


# #### Decision Tree 

# In[47]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
model_acc(dt)


# #### Random Forest

# In[48]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
model_acc(rf)


# ### Train Model, Predict, and Calculating Model Accuracy

# ###### Here, PCA transformed data set was trained using Logistic regression, Decision tree, and Random forest. Then, Test set was used to predict y values and calculate accuracy for each models. from that best model can be choosen. Here we have to study accuracy , confution matrix   and classification report.

# In[49]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
models = [
    ('Logistic Regression', LogisticRegression()),
    ('Decision Tree', DecisionTreeClassifier()),
    ('Random Forest', RandomForestClassifier())
]

for model_name, model in models:
    
    model.fit(x_train_pca,y_train)
    
    y_train_pred = model.predict(x_train_pca)
    y_test_pred = model.predict(x_test_pca)
    
    train_accuracy = accuracy_score(y_train,y_train_pred)
    test_accuracy = accuracy_score(y_test,y_test_pred)
    conf_matrix = confusion_matrix(y_test,y_test_pred)
    class_report = classification_report(y_test, y_test_pred)
    
    print(model_name)
    print('Train_accuracy :',train_accuracy)
    print('Test_accuracy :\n',test_accuracy)
    print('Confusion_matrix :\n',conf_matrix)
    print('Classification_report :\n',class_report)


# ### Train Model, Predict, and Calculating Model Accuracy for scaled data (without PCA transformed)

# In[50]:


for model_name, model in models:
    
    model.fit(x_train_scaled,y_train)
    
    y_train_pred = model.predict(x_train_scaled)
    y_test_pred = model.predict(x_test_scaled)
    
    train_accuracy = accuracy_score(y_train,y_train_pred)
    test_accuracy = accuracy_score(y_test,y_test_pred)
    conf_matrix = confusion_matrix(y_test,y_test_pred)
    class_report = classification_report(y_test, y_test_pred)
    
    print(model_name)
    print('Train_accuracy :',train_accuracy)
    print('Test_accuracy :\n',test_accuracy)
    print('Confusion_matrix :\n',conf_matrix)
    print('Classification_report :\n',class_report)


# #### Train Model, Predict, and Calculating Model Accuracy (Without scaled & PCA transformed)

# In[51]:


for model_name, model in models:
    
    model.fit(x_train,y_train)
    
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    
    train_accuracy = accuracy_score(y_train,y_train_pred)
    test_accuracy = accuracy_score(y_test,y_test_pred)
    conf_matrix = confusion_matrix(y_test,y_test_pred)
    class_report = classification_report(y_test, y_test_pred)
    
    print(model_name)
    print('Train_accuracy :',train_accuracy)
    print('Test_accuracy :\n',test_accuracy)
    print('Confusion_matrix :\n',conf_matrix)
    print('Classification_report :\n',class_report)


# ###### By studing accuracy factors, best model was given by logistic regression with scaled and PCA transformed data. It's accuracy is about 85.95% and precision is also high.


# #### Logistic Regression

# In[46]:


from sklearn.linear_model import LogisticRegression
lo = LogisticRegression()
model_acc(lo)


# #### Decision Tree 

# In[47]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
model_acc(dt)


# #### Random Forest

# In[48]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
model_acc(rf)


# ### Train Model, Predict, and Calculating Model Accuracy

# ###### Here, PCA transformed data set was trained using Logistic regression, Decision tree, and Random forest. Then, Test set was used to predict y values and calculate accuracy for each models. from that best model can be choosen. Here we have to study accuracy , confution matrix   and classification report.

# In[49]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
models = [
    ('Logistic Regression', LogisticRegression()),
    ('Decision Tree', DecisionTreeClassifier()),
    ('Random Forest', RandomForestClassifier())
]

for model_name, model in models:
    
    model.fit(x_train_pca,y_train)
    
    y_train_pred = model.predict(x_train_pca)
    y_test_pred = model.predict(x_test_pca)
    
    train_accuracy = accuracy_score(y_train,y_train_pred)
    test_accuracy = accuracy_score(y_test,y_test_pred)
    conf_matrix = confusion_matrix(y_test,y_test_pred)
    class_report = classification_report(y_test, y_test_pred)
    
    print(model_name)
    print('Train_accuracy :',train_accuracy)
    print('Test_accuracy :\n',test_accuracy)
    print('Confusion_matrix :\n',conf_matrix)
    print('Classification_report :\n',class_report)


# ### Train Model, Predict, and Calculating Model Accuracy for scaled data (without PCA transformed)

# In[50]:


for model_name, model in models:
    
    model.fit(x_train_scaled,y_train)
    
    y_train_pred = model.predict(x_train_scaled)
    y_test_pred = model.predict(x_test_scaled)
    
    train_accuracy = accuracy_score(y_train,y_train_pred)
    test_accuracy = accuracy_score(y_test,y_test_pred)
    conf_matrix = confusion_matrix(y_test,y_test_pred)
    class_report = classification_report(y_test, y_test_pred)
    
    print(model_name)
    print('Train_accuracy :',train_accuracy)
    print('Test_accuracy :\n',test_accuracy)
    print('Confusion_matrix :\n',conf_matrix)
    print('Classification_report :\n',class_report)


# #### Train Model, Predict, and Calculating Model Accuracy (Without scaled & PCA transformed)

# In[51]:


for model_name, model in models:
    
    model.fit(x_train,y_train)
    
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    
    train_accuracy = accuracy_score(y_train,y_train_pred)
    test_accuracy = accuracy_score(y_test,y_test_pred)
    conf_matrix = confusion_matrix(y_test,y_test_pred)
    class_report = classification_report(y_test, y_test_pred)
    
    print(model_name)
    print('Train_accuracy :',train_accuracy)
    print('Test_accuracy :\n',test_accuracy)
    print('Confusion_matrix :\n',conf_matrix)
    print('Classification_report :\n',class_report)


# ###### By studing accuracy factors, best model was given by logistic regression with scaled and PCA transformed data. It's accuracy is about 85.95% and precision is also high.