#!/usr/bin/env python
# coding: utf-8

# ## Section 1 - Data manipulation
# 
# 

# Firstly, we will upload the required libraries for this task.

# In[35]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import xgboost
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, plot_confusion_matrix, plot_precision_recall_curve
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from chart_studio import plotly
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
import plotly.figure_factory as ff
# Using plotly + cufflinks in offline mode
import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)
import plotly.io as pio


# In[36]:


data_full = pd.read_csv('/Users/m.nicajus/Downloads/bank/bank-full.csv', sep = ';') #will be used as train data 
data_test = pd.read_csv('/Users/m.nicajus/Downloads/bank/bank.csv', sep = ';') #will be used as test data 


# In[37]:


#This is full data of bank
data_full


# In[38]:


data_test


# By checking the types of the data for each column we get the following.

# In[39]:


data_full['age'].value_counts()
print(data_full.dtypes)


# We are checking in this data set whether there are NaN values for each columns

# In[40]:


data_full.isnull().sum()


# In[41]:


data_test.isnull().sum()


# From the above we can say that there are no non-available values in this data set. By the following line of code we can check if there is a duplications in the row of this data set.

# In[42]:


print(data_full.duplicated().value_counts())
print(data_test.duplicated().value_counts())


# From the output above we can conclude that there ar no duplicated values in given data sets. There were no rows that were needed to be removed. We can identify each row of this data set unique. 

# Furthermore, variables in this data set need to be checked to assess the frequency of each entry of the column. Consider the following:

# In[43]:


for col in data_full.columns.values:
    print(f"\033[1m{col} \n{20 * '-'}\033[0m")
    print(data_full[col].value_counts(), '\n')


# In[44]:


#Same we can do for the test bank data
for col in data_test.columns.values:
    print(f"\033[1m{col} \n{20 * '-'}\033[0m")
    print(data_full[col].value_counts(), '\n')


# Note that we have potential 16 features that need to be assessed. We will use summary statistics to describe numerical and categorical columns of data_full set.

# In[45]:


data_full.describe()


# In[46]:


data_full.describe(include = ['O']) #this is summary statistics for categorical features


# Note that we have four categorical variables as follows: default, housing, loan, y. These features can be converted into boolean variables in order to make further computations meaning that it return 0 if value in column_name is 'no', returns 1 if value in column_name is 'yes'. 

# In[47]:


def bool_yn(row, column_name):
    return 1 if row[column_name] == 'yes' else 0

def clean_data(data_full):
    
    cleaned_data_f = data_full.copy()
    
    
    bool_columns = ['default', 'housing', 'loan', 'y']
    for bool_col in bool_columns:
        cleaned_data_f[bool_col] = data_full.apply(lambda row: bool_yn(row, bool_col),axis=1)
        
    return cleaned_data_f
  


# In[48]:


data_full_cleaned = clean_data(data_full)
data_full_cleaned


# In[49]:


#Let's make a visualize about our target column ('y' column)
plt.figure(figsize=(12,6))
labels = ['No','Yes']
data_full['y'].value_counts().plot.pie(shadow=True,
                                labels=labels,
                                autopct='%0.f%%',
                                explode = [0.0,0.2])
plt.title('The visualisation of the term deposit subscription')
plt.legend()


# Note that new variables can be created in order to have a better understanding of given data. We will use numerical columns to inspect the effect of the term deposit subscription. We are doing this to verify there is a need to remove some variables for further analysis. Let us define the number of contacts performed during this campaign and term deposit.Consider the following:

# In[50]:


campaign_y_df = pd.DataFrame() # empty data frame
campaign_y_df['campaign_yes'] = (data_full_cleaned[data_full_cleaned['y'] == 1][['y','campaign']].describe())['campaign']
campaign_y_df['campaign_no'] = (data_full_cleaned[data_full_cleaned['y'] == 0][['y','campaign']].describe())['campaign']

campaign_y_df


# Same we can do with same with column 'age' to see the corrrespondend values of the summary statistics as follows:

# In[51]:


age_y_df = pd.DataFrame() # empty data frame
age_y_df['age_yes'] = (data_full_cleaned[data_full_cleaned['y'] == 1][['y','age']].describe())['age']
age_y_df['age_no'] = (data_full_cleaned[data_full_cleaned['y'] == 0][['y','age']].describe())['age']

age_y_df


# Furthermore, for the 'balance' column we have the following:

# In[52]:


balance_y_df = pd.DataFrame() # empty data frame
balance_y_df['balance_yes'] = (data_full_cleaned[data_full_cleaned['y'] == 1][['y','balance']].describe())['balance']
balance_y_df['balance_no'] = (data_full_cleaned[data_full_cleaned['y'] == 0][['y','balance']].describe())['balance']

balance_y_df


# And with the column 'duration' we have the following

# In[53]:


duration_y_df = pd.DataFrame() # empty data frame
duration_y_df['duration_yes'] = (data_full_cleaned[data_full_cleaned['y'] == 1][['y','duration']].describe())['duration']
duration_y_df['duration_no'] = (data_full_cleaned[data_full_cleaned['y'] == 0][['y','duration']].describe())['duration']

duration_y_df


# For the column 'previous' we have:

# In[54]:


previous_y_df = pd.DataFrame() # empty data frame
previous_y_df['previous_yes'] = (data_full_cleaned[data_full_cleaned['y'] == 1][['y','previous']].describe())['previous']
previous_y_df['previous_no'] = (data_full_cleaned[data_full_cleaned['y'] == 0][['y','previous']].describe())['previous']

previous_y_df


# In[55]:


pdays_y_df = pd.DataFrame() # empty data frame
pdays_y_df['pdays_yes'] = (data_full_cleaned[data_full_cleaned['y'] == 1][['y','pdays']].describe())['pdays']
pdays_y_df['pdays_no'] = (data_full_cleaned[data_full_cleaned['y'] == 0][['y','pdays']].describe())['pdays']

pdays_y_df


# ***
# From the above calculations we can conclude the following statements:
# 
# * The number of contacts with term deposit subcription was less than those with not having term deposit subscription. This means that people with term deposit product were less contacted than those with no product( bank term deposit) and they are more willing to include the deposit of money into an account at a financial institution. 
# 
# * Older people tend to be subscribed for bank term deposit.
# 
# * People who are subscribed to a term deposit are tend to have greater balances than those who are not. That is obvious, because they need to have higher or sufficient funds in order to be subscripted to use given financial product.
# 
# * Duration of the people of contact time is longer for those who are subscribed to term deposit than those who are not subscribed to this financial product. This is due to potential updates of the current product or explanations of the product(answering questions) to the subscribers of the term deposit.
# 
# * The average number of contacts performed before the current campaign were higher for those people who are not having a term deposit subscription.

# By checking the summary statistics of the feature variable 'pdays', we are dropping this column and consider the following data

# In[56]:


data_full_cleaned= data_full_cleaned.drop(columns = ['pdays'])


# In[57]:


data_full_cleaned


# Furthermore, we are going to assess the categorical variables as dummy variables. These will be used in the further analysis.Consider the following:

# In[58]:


categorical_col = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']

data_full_cleaned_with_dummies = pd.concat([data_full_cleaned.drop(categorical_col, axis=1),
                                     pd.get_dummies(data_full_cleaned[categorical_col], prefix=categorical_col, prefix_sep='_',
                                     drop_first=True, dummy_na=False)], axis=1)
    
data_full_cleaned_with_dummies


# ## Section 2 - Data visualisation

# In this section weare going to viusalise the data set in order to identify potential outliers or to get better insight of the data. Note that data will be modified based on the graphs.

# In[59]:


job_fig = data_full_cleaned['job'].iplot(kind='hist', xTitle='jobs',
                  yTitle='count', title='Distribution of the jobs')
job_fig.show(renderer = 'chrome')


# In[60]:


data_full_cleaned['marital'].iplot(kind='hist', xTitle='Marital status',
                  yTitle='count', title='Distribution of the marital status', color = 'green')


# In[61]:


data_full_cleaned['education'].iplot(kind='hist', xTitle='Education',
                  yTitle='count', title='Distribution of the types of education', color = 'purple')


# In[62]:


data_full_cleaned['contact'].iplot(kind='hist', xTitle='Types of contacting person',
                  yTitle='count', title='Distribution of the ways of contacting clients', color = 'blue')


# In[63]:


data_full_cleaned['month'].iplot(kind='hist', xTitle='Months where the person has been contacted',
                  yTitle='count', title='Distribution of the months where the person has been contacted', color = 'pink')


# In[64]:


data_full_cleaned['poutcome'].iplot(kind='hist', xTitle='poutcome',
                  yTitle='count', title='Distribution of the poutcome', color = 'navy')


# Let's see how term deposit column 'y' value varies depending on various categorical column values. Consider the following function showing the required outcome.

# In[65]:


temp_m = pd.DataFrame()
temp_m['No_deposit'] = data_full_cleaned[data_full_cleaned['y'] == 0]['marital'].value_counts()
temp_m['Yes_deposit'] = data_full_cleaned[data_full_cleaned['y'] == 1]['marital'].value_counts()


# In[66]:


temp_m.iplot(kind='bar', xTitle='marital status',
                  yTitle='count', title='Distribution of the poutcome')


# In[67]:


temp_j = pd.DataFrame()
temp_j['No_deposit'] = data_full_cleaned[data_full_cleaned['y'] == 0]['job'].value_counts()
temp_j['Yes_deposit'] = data_full_cleaned[data_full_cleaned['y'] == 1]['job'].value_counts()


# In[68]:


temp_j.iplot(kind='bar', xTitle='Title of job',
                  yTitle='count', title='Distribution of the job based on term deposit subscription participants')


# In[69]:


temp_e = pd.DataFrame() # for education column
temp_e['No_deposit'] = data_full_cleaned[data_full_cleaned['y'] == 0]['education'].value_counts()
temp_e['Yes_deposit'] = data_full_cleaned[data_full_cleaned['y'] == 1]['education'].value_counts()


# In[70]:


temp_e.iplot(kind='bar', xTitle='Type of education',
                  yTitle='count', title='Distribution of the education based on term deposit subscription participants')


# In[71]:


temp_c = pd.DataFrame() # for contact column
temp_c['No_deposit'] = data_full_cleaned[data_full_cleaned['y'] == 0]['contact'].value_counts()
temp_c['Yes_deposit'] = data_full_cleaned[data_full_cleaned['y'] == 1]['contact'].value_counts()


# In[72]:


temp_c.iplot(kind='bar', xTitle='Type of contacting clients',
                  yTitle='count', title='Distribution of the ways of contacting clients based on the response of the term deposit subscription ')


# In[73]:


temp_mth = pd.DataFrame() # for months column
temp_mth['No_deposit'] = data_full_cleaned[data_full_cleaned['y'] == 0]['month'].value_counts()
temp_mth['Yes_deposit'] = data_full_cleaned[data_full_cleaned['y'] == 1]['month'].value_counts()


# In[74]:


temp_mth.iplot(kind='bar', xTitle='Month',
                  yTitle='count', title='Distribution of the months based on the response of the term deposit subscription ')


# We are going to take a look which most common job among the yes said clients and what is the percent of this job to all jobs. We will apply this method for the rest  categorical variables

# ***
# # Job feature

# In[75]:


data_full_cleaned[['job','y']].groupby(['job'], as_index=True).mean().sort_values(
    by='y',ascending=False).style.background_gradient(axis=None
                                        , low=0.75, high=1.0)


# From this we can say that students were the most active in terms of being subscribed to this financial product. The least were blue-collar job position people. However, we are going to count the amount of people who were interested in subscribing term deposit product or not. Consider the following lines of code:

# In[76]:


pd.crosstab(data_full_cleaned['job'],data_full_cleaned['y'], margins=True, margins_name='Total').style.background_gradient(axis=None
                                                                                  , low=0.75, high=1.0)


# By inspecting the amounts of each job class participants who are selective between subscribing financial product or not subscribing term deposit we can say that there is a enormous gap between students and blue-collar job class. However, we can make conclusion that 
# class are not interested joining the deposit whereas other classes like students, retired are more interested to subscribe.

# ***
# # Marital feature

# In[77]:


data_full_cleaned[['marital','y']].groupby(['marital'], as_index=True).mean().sort_values(
    by='y',ascending=False).style.background_gradient(axis=None
                                        , low=0.75, high=1.0)


# In[78]:


pd.crosstab(data_full_cleaned['marital'],data_full_cleaned['y'], margins=True, margins_name='Total').style.background_gradient(axis=None
                                                                                  , low=0.75, high=1.0)


# From this we can say that single people are the most tempted to participate on term deposit subcription in proporion. However, the married people were participated more than the single more than double creating a huge gap between these 2 marital status

# ***
# # Education feature

# In[79]:


data_full_cleaned[['education','y']].groupby(['education'], as_index=True).mean().sort_values(
    by='y',ascending=False).style.background_gradient(axis=None
                                        , low=0.75, high=1.0)


# In[80]:


pd.crosstab(data_full_cleaned['education'],data_full_cleaned['y'], margins=True, margins_name='Total').style.background_gradient(axis=None
                                                                                  , low=0.75, high=1.0)


# ***
# # Month  feature

# In[81]:


data_full_cleaned[['month','y']].groupby(['month'], as_index=True).mean().sort_values(
    by='y',ascending=False).style.background_gradient(axis=None
                                        , low=0.75, high=1.0)


# In[82]:


pd.crosstab(data_full_cleaned['month'],data_full_cleaned['y'], margins=True, margins_name='Total').style.background_gradient(axis=None
                                                                                  , low=0.75, high=1.0)


# ***
# # Contact  feature

# In[83]:


data_full_cleaned[['contact','y']].groupby(['contact'], as_index=True).mean().sort_values(
    by='y',ascending=False).style.background_gradient(axis=None
                                        , low=0.75, high=1.0)


# In[84]:


pd.crosstab(data_full_cleaned['contact'],data_full_cleaned['y'], margins=True, margins_name='Total').style.background_gradient(axis=None
                                                                                  , low=0.75, high=1.0)


# In[85]:


gde = sns.FacetGrid(data=data_full_cleaned, col='y',row='month',size=2.2, aspect=1.6)
gde.map(sns.histplot, 'age', alpha=.8, bins=12)
gde.add_legend()


# Moreover, we are going to visualise and gain more insight for numerical columns of the given data set.

# In[86]:


num_columns = ['age', 'balance', 'day','duration', 'campaign', 'previous']

fig, axs = plt.subplots(2, 3, sharex=False, sharey=False, figsize=(20, 15))

counter = 0
for num_column in num_columns:
    
    trace_x = counter // 3
    trace_y = counter % 3
    
    axs[trace_x, trace_y].hist(data_full_cleaned[num_column], color = 'yellow')
    
    axs[trace_x, trace_y].set_title(num_column)
    
    counter += 1

plt.show()


# Now we are going to assess each given numerical feature as follows:

# ***
# # Age  feature

# In[87]:


sns.distplot(data_full_cleaned['age'],fit=stats.norm, color = 'olive')
vis = plt.figure()
res = stats.probplot(data_full_cleaned['age'], plot=plt)


# In[88]:


g = sns.FacetGrid(data_full_cleaned, col='y')
g.map(plt.hist, 'age', bins=20)


# ***
# # Balance  feature

# In[89]:


griddata = sns.FacetGrid(data_full_cleaned, hue='y', size=4,aspect=2.5)
griddata.map(sns.kdeplot, 'balance', shade=True, alpha=.6)
griddata.set(xlim=(data_full_cleaned['balance'].min(),data_full_cleaned['balance'].max()))
griddata.add_legend()


# In[90]:


data_full_cleaned.corr().balance.y # current correlation


# By making into 4 ranges of 'balance' column, we can improve correlation between 'balance' and 'y'

# In[91]:


data_full_cleaned['Balance_band'] = pd.cut(data_full_cleaned['balance'],4)
data_full_cleaned[['Balance_band', 'y']].groupby('Balance_band', as_index=True).mean().sort_values(by='Balance_band', ascending=False)


# In[92]:


# By making into 3 categories we find the following 
for bal_col in[data_full_cleaned]:
    bal_col.loc[ bal_col['balance'] <= 0, 'balance'] = 0
    bal_col.loc[ (bal_col['balance'] > 0) & (bal_col['balance'] <= 74590.5), 'balance'] = 1
    bal_col.loc[ (bal_col['balance'] > 74590.5), 'balance'] = 2


# In[93]:


data_full_cleaned.corr().balance.y # new correlation after modification


# In[94]:


data_full_cleaned['balance'].describe()


# # Duration feature

# In[95]:


sns.distplot(data_full_cleaned['duration'],fit=stats.norm)
fig = plt.figure()
res = stats.probplot(data_full_cleaned['duration'], plot=plt)


# In[96]:


data_full_cleaned['dband'] = pd.cut(data_full_cleaned['duration'],4)
data_full_cleaned[['dband','y']].groupby('dband', as_index=True).mean().sort_values(by='y', ascending=False)


# In[97]:


data_full_cleaned = data_full_cleaned.drop('Balance_band', axis=1)


# In[98]:


data_full_cleaned


# # Day Feature 

# In[99]:


pd.crosstab(data_full_cleaned['day'],data_full_cleaned['y'], margins=True).sort_values(by=1, ascending=False).style.background_gradient(axis=None
                                                                                  , low=0.75, high=1.0)


# In[100]:


gdday = sns.FacetGrid(data=data_full_cleaned, row='day', size=2.2, aspect=1.6)
gdday.map(sns.kdeplot, 'y',shade=True, alpha=.6)
gdday.add_legend()


# # Campaign feature

# In[101]:


pd.crosstab(data_full_cleaned['campaign'],data_full_cleaned['y'], margins=True).sort_values(by=1, ascending=False).style.background_gradient(axis=None
                                                                                  , low=0.75, high=1.0)


# In[102]:


gdcamp = sns.FacetGrid(data=data_full_cleaned, row='campaign', size=2.2, aspect=1.6)
gdcamp.map(sns.kdeplot, 'y',shade=True, alpha=.6)
gdcamp.add_legend()


# In[103]:


data_full_cleaned['camp_split'] = pd.cut(data_full_cleaned['campaign'],2)


# In[104]:


data_full_cleaned[['camp_split','y']].groupby('camp_split', as_index=True).mean().sort_values(by='y', ascending=False)


# In[105]:


data_full_cleaned.drop('camp_split', axis=1, inplace=True)


# We are going to perform the correlation matrix to verify if there is a correlation between numerical variables and target variable 'y'. Consider the following

# In[106]:


corrs = data_full_cleaned.corr()
figure = ff.create_annotated_heatmap(
    z=corrs.values,
    x=list(corrs.columns),
    y=list(corrs.index),
    annotation_text=corrs.round(2).values,
    showscale=True)
figure


# From the corellation matrix above we see that the duration column has the highest correlation with term deposit 'y' in the whole correlation matrix with 0.39
# 
# We will perform the same with the dummy variables where they were defined before. Consider the following:

# In[107]:


corrs_with_dummies = data_full_cleaned_with_dummies.corr()
figure_with_dummies = ff.create_annotated_heatmap(
    z=corrs_with_dummies.values,
    x=list(corrs_with_dummies.columns),
    y=list(corrs_with_dummies.index),
    annotation_text=corrs_with_dummies.round(2).values,
    showscale=True)
figure_with_dummies


# Since we have a large correlation matrix we will define a function which returns highest correlation values of this data set with dummie variables. We have the following:

# In[108]:


def corrFilter(x: pd.DataFrame, bound: float):
    xCorr = data_full_cleaned_with_dummies.corr()
    xFiltered = xCorr[(xCorr >= bound) & (xCorr !=1)]
    xFlattened = xFiltered.unstack().sort_values().drop_duplicates()
    return xFlattened

corrFilter(data_full_cleaned_with_dummies, .4).dropna()


# These correlations from the correlation matrix are the most positive correlations in this data set. In the following lines we are chekcing the amount of the positive and negative entries in the correlation matrix.

# In[116]:


neg_cor = corrs_with_dummies[corrs_with_dummies <0]
neg_cor.count().sum() # for negative correlations


# In[117]:


pos_cor = corrs_with_dummies[corrs_with_dummies >0]
pos_cor.count().sum() # for positive correlations


# # 3 Section - Data Modelling

# In[118]:


# Independnet features
X_ind = data_full_cleaned_with_dummies.drop(['y'], axis=1)
# Dependent feature
y_d = data_full_cleaned_with_dummies['y']
X_ind.head()


# In[120]:


X_train,X_test,y_train,y_test= train_test_split(X_ind,y_d,test_size=0.20,random_state=42)

X_train = pd.DataFrame(X_train, columns = X_ind.columns)
X_test = pd.DataFrame(X_test, columns=X_ind.columns)


# In[121]:


#Logistic regression model
logit= LogisticRegression()
logit.fit(X_train, y_train)
print(logit.fit(X_train, y_train))


# In[122]:


# Predicting the model
pred_logit= logit.predict(X_test)


# In[123]:


print("The accuracy of logit model is:", accuracy_score(y_test, pred_logit))
print(classification_report(y_test, pred_logit))


# In[124]:


plot_precision_recall_curve(logit,X_test,y_test)


# In[125]:


plot_confusion_matrix(logit, X_test, y_test, cmap="Greens")


# In[126]:


logit.predict_proba(X_test)


# We will scale the features in order our model would not lead to bias towards higher range of values(outliers). Consider the following lines of code:

# In[128]:


scaler= StandardScaler()
X_sc= scaler.fit_transform(X_ind)


# In[129]:


X_sc


# In[130]:


X_train_sc,X_test_sc,y_train_sc,y_test_sc= train_test_split(X_sc,y_d,test_size=0.20,random_state=42)


# In[131]:


logit_sc= LogisticRegression()
logit_sc.fit(X_train_sc, y_train_sc)
print(logit.fit(X_train_sc, y_train_sc))


# In[132]:


# Predicting the model
pred_logit_sc= logit.predict(X_test_sc)


# In[133]:


print("The accuracy of logit model is:", accuracy_score(y_test_sc, pred_logit_sc))
print(classification_report(y_test_sc, pred_logit_sc))


# In[134]:


plot_precision_recall_curve(logit_sc,X_test_sc,y_test_sc)


# In[135]:


plot_confusion_matrix(logit_sc, X_test_sc, y_test_sc, cmap="Reds")


# In[136]:


logit_sc.predict_proba(X_test_sc)


# There is another way of performing Logistic regression model as follows:

# In[137]:


# Train the logistic regression model on the training data
logistic_reg = LogisticRegression(solver='lbfgs').fit(X_train, np.ravel(y_train))

# Create predictions of probability for loan status using test data
preds = logistic_reg.predict_proba(X_test)

# Create dataframes of first five predictions, and first five true labels
preds_df = pd.DataFrame(preds[:,1][0:5], columns = ['prob_deposit'])
true_df = y_test.head()

# Concatenate and print the two data frames for comparison
print(pd.concat([true_df.reset_index(drop = True), preds_df], axis = 1))


# In[138]:


# Create predictions and store them in a variable
preds = logistic_reg.predict_proba(X_test)

# Print the accuracy score the model
print(logistic_reg.score(X_test, y_test))


# We are going to visualise receiver operating characteristic curve (ROC curve) of the probabilities of the term deposit subscription. Consider the following

# In[139]:


prob_term_deposit = preds[:, 1]
fallout, sensitivity, thresholds = metrics.roc_curve(y_test, prob_term_deposit)
plt.plot(fallout, sensitivity, color = 'darkorange')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.show()


# In[140]:


auc = roc_auc_score(y_test, prob_term_deposit)
auc


# In[142]:


#Finding the accuracy using cross validation method
score_log_reg = cross_val_score(logit, X_ind, y_d, scoring='accuracy', cv=10)
print(score_log_reg)
print(score_log_reg.mean())


# This gives smaller accuracy of the model usinig cross validation method rather than using classification report analysis

# We will see another way to see the correlation using logistic regression model, this time we use scaled features in order to have more precise score.

# # Extra part: Assesing other models for predicting target variable

# We will use other models just to see how they differ from logistic model just for curiosity. Firstly, we are going to assess the XGBoost classifier model. Consider the following:

# In[143]:


xgb = xgboost.XGBClassifier(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)
xgb.fit(X_train,y_train.squeeze().values)


# In[144]:


y_train_preds = xgb.predict(X_train)
y_test_preds = xgb.predict(X_test)

print('XGB accuracy score for train: %.3f: test: %.3f' % (
        accuracy_score(y_train, y_train_preds),
        accuracy_score(y_test, y_test_preds)))


# Furthermore, we are going to assess the importance of the features in this data set using XGBoost model. We have the following:

# In[145]:


head = ["name", "score"]
values = sorted(zip(X_train.columns, xgb.feature_importances_), key=lambda x: x[1] * -1)
xgb_feature_importances = pd.DataFrame(values, columns = head)

#plot feature importances
x_pos = np.arange(0, len(xgb_feature_importances))
plt.bar(x_pos, xgb_feature_importances['score'])
plt.xticks(x_pos, xgb_feature_importances['name'])
plt.xticks(rotation=90)
plt.title('Feature importances (XGB)')

plt.show()


# From the graph above we can see that by showing the importance feature including dummy variables, most important features are :
# 
# * The last contact duration of the client 
# * The successful outcome of the previous marketing campaign
# * The unknown type of communication for promoting a campaign(Probably was live meeting or proposing the campaign in person)

# Another model which we will assess will be Support vector machine(SVM). Consider the following analysis:

# In[147]:


svc = svm.SVC(kernel='rbf', C=70, gamma=0.001).fit(X_train_sc,y_train_sc)
prediction_svm = svc.predict(X_test_sc)
print(classification_report(y_test_sc,prediction_svm))


# In[148]:


percentage = svc.score(X_test_sc,y_test_sc)
percentage    


# In[149]:


plot_confusion_matrix(svc, X_test_sc, y_test_sc, cmap="PuBu")

