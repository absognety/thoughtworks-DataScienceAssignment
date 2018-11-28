def intro():

    """
    Thoughtworks Data Science Assignment
    
    Context
    In Banking industry, loan applications are generally approved after a 
    thorough background check of the customer's repayment capabilities. 
    Credit Score plays a significant role in identifying customer's financial
    behavior (specifically default). However, people belonging to rural India 
    don't have credit score and it is difficult to do a direct assessment.
    The accompanying file trainingData.csv contains some of the information 
    that is collected for loan applications of rural customers. 
    We need to understand the maximum repayment capability of customers 
    which can be used to grant them the desired amount.
    
    Description of variables:
    • Id: Primary Key
    • Personal Details: city, age, sex, social_class
    • Financial Details: primary_business, secondary_business, annual_income, 
    monthly_expenses, old_dependents, young_dependents
    • House Details: home_ownership, type_of_house, occupants_count, 
    house_area, sanitary_availability, water_availability
    • Loan Details: loan_purpose, loan_tenure, loan_installments, 
    loan_amount (these contain loan details of loans that have been 
    previously given, and which have been repaid)
    
    """
    
    pass

import os
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression
import operator

file_path = 'C:\\Users\\cvikas10\\Documents\\thoughtworks_DataScience\\Data_Analysis_Exercise'

loanData = pd.read_csv(os.path.join(file_path,'trainingData.csv'),
                       converters={'Id':str,	
                       'city':str,'age':int,'sex':str,	
                       'social_class':str,	
                       'primary_business':str,	
                       'secondary_business':str,	
                       'annual_income':int,	
                       'monthly_expenses':str,	
                       'old_dependents':int,	
                       'young_dependents':int,	
                       'home_ownership':str,	
                       'type_of_house':str,	
                       'occupants_count':int,	
                       'house_area':int,	
                       'sanitary_availability':str,	
                       'water_availabity':str,	
                       'loan_purpose':str,	
                       'loan_tenure':int,	
                       'loan_installments':int,	
                       'loan_amount':int})

print (loanData.describe())
print (loanData.info())
print (loanData.dtypes.unique())
categorical_cols = loanData.select_dtypes(include=['O']).columns.tolist()
categorical_cols.remove('Id')
categorical_cols.remove('monthly_expenses')

#Missing Data entries row-wise and column-wise
print (loanData.isnull().any(axis=1).sum(),'/',len(loanData.columns))
print (loanData.isnull().any(axis=0).sum(),'/',len(loanData))

#category counts of factor variables
def value_cnts(cols,data):
    factor_cnts = {}
    for col in cols:
        cnts = data[col].value_counts()
        factor_cnts[col] = cnts
    return factor_cnts
    
print (value_cnts(categorical_cols,loanData))


#ANOVA F-test for Correlation between factor variables
def anova_ftest(cols,data):
    fpvalues = {}
    for col in cols:
        model = smf.ols(formula='loan_amount ~ '+col, data=data)
        results = model.fit()
        fvalue = results.fvalue
        pvalue = results.f_pvalue
        rsqvalue = results.rsquared_adj
        fpvalues[col] = (fvalue,pvalue,rsqvalue)
    return (fpvalues)
    
FP_values = anova_ftest(categorical_cols,loanData)
FP_df = pd.DataFrame(FP_values,columns=list(FP_values.keys()),
                     index=['F_value','P_value','RSq_value'])

values = FP_df.loc[['F_value','P_value'],:].T
print (values)

#By p-value and F-statistic
filter_FP = values[(values['F_value']>20) & (values['P_value']<0.05)]
print (filter_FP)

factor_imp_cols = filter_FP.index.tolist()


#Excluding outliers in Age
ages = np.array(loanData['age'])
age_min = np.min(ages) #age_min = 2
age_max = np.max(ages) #age_max = 766105

rem_ages = ages[ages > 100]

data = loanData[loanData['age'] < 100]
data = data[data['age'] != 2]
#Total 5 records are deleted
#Total number of data points considered now = 39995

#Excluding data points having no primary business and loan purpose
# Number of records having null or N/A values = 26
#26/39995 ~ 0.0006500812601575197 ~ 0.06%
#ignoring the above data points

data = data[(data['primary_business']!='') & (data['loan_purpose']!='#N/A')]
#Total number of data points under consideration now = 39969.

#filter on annual_income where it is equal to 0
#loan approval depends on financial support of rural citizen.
#also putting filter on monthly_expenses where it is NA

rem_dt_ = data[data['monthly_expenses'] == 'NA']
rem_dt = data[(data['annual_income'] == 0) & (data['monthly_expenses'] == 'NA')]
#both of them have same records, so all the records
#having annual income as 0 also have monthly_expenses as NA
print (rem_dt_.shape) # (120,21)
print (rem_dt.shape) #(120,21)

null_content = value_cnts(categorical_cols,rem_dt)
print (null_content)

def null_analysis():
    
    """
    home_ownership ~ 106/120 = 88.33% NULL
    primary_business ~ too biased (school = 103/120 = 85.83%)
    sanitary_availability ~ 106/120 = 88.33% NULL
    secondary_business ~ 119/120 = 99.16% NULL
    social_class ~ 120/120 = 100% NULL
    type_of_house ~ 108/120 = 90% NULL
    water_availabity ~ 120/120 = 100% NULL
    ignoring these data points as most of columns have null content
    
    """
    pass

data = data[~data.index.isin(rem_dt.index)]
data['monthly_expenses'] = data['monthly_expenses'].astype(int)

#Taking numeric variables for correlation analysis
#correlation analysis with target variable: loan_amount
numeric_cols = data.select_dtypes(include=['int64','int32']).columns.tolist()
corr_matrix = data[numeric_cols].corr()
corr_target = corr_matrix['loan_amount']
print (corr_target)

#weak correlation coefficients for all numeric variables.
_data = data[(data['water_availabity']=='NULL') & (data['sanitary_availability']=='NULL')]
data = data[~data.index.isin(_data.index)]

null_content2 = value_cnts(categorical_cols,_data)

def null_content2_analysis():
    
    """
    Except primary_business, loan_purpose and city
    remaining variables like:
        social_class ~ 100% NULL (102/102)
        sex ~ F (100%) ~ Too biased
        secondary_business ~ 100% NULL (102/102)
        water_availabity ~ 100% NULL (102/102)
        sanitary_availability ~ 100% NULL (102/102)
    have maximum percentage of NULL's
    ignoring these data records considering above information.
    
    """
    pass

#Excluding outlier in occupants_count column
max_occupcnt = np.max(data['occupants_count'])
sns.boxplot(data['occupants_count'])
plt.show()
#max occupant count = 950000 which doesn't make sense.
outlier_occup = data[data['occupants_count']==max_occupcnt]
data = data[~data.index.isin(outlier_occup.index)]

###Encoding factor/character variables to numeric values using frequency values
char_df = data[factor_imp_cols]
df = char_df
def category_num(categs,col,data):
    for cat in categs:
        numerator = data[data[col]==cat].shape[0]
        denominator = data[col].shape[0]
        score = numerator/denominator
        data[col] = data[col].replace(cat,score)
    return

def factor_to_num(cols,data):
    for c in cols:
        cats = pd.Categorical(data[c]).categories
        category_num(cats,c,data)
    return data

char_df1 = factor_to_num(factor_imp_cols,df)

def prepare_data(charDF):
    char_df_num = charDF.reset_index(drop=True)
    
    numeric_df = data[numeric_cols].iloc[:,:-1]
    numeric_df_num = numeric_df.reset_index(drop=True)
    y = data.iloc[:,-1]
    y_num = y.reset_index(drop=True)
    processed_data = pd.concat([char_df_num,numeric_df_num],axis=1)
    
    return processed_data,y_num,numeric_df_num,char_df_num

processed_data,y_num,numeric_df_num,char_df_num = prepare_data(char_df1)

def run_models(processed_dt,Y):
        
    trainX,testX,trainY,testY = train_test_split(processed_dt,Y,test_size=0.30, 
                                                 random_state=42,
                                                 shuffle=True)
            
    print (trainX.shape)
    print (trainY.shape)
    print (testX.shape)
    print (testY.shape)
    params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
              'learning_rate': 0.01, 'loss': 'ls'}
    model = GradientBoostingRegressor(**params)
    
    adaB_params = {'n_estimators':500,'learning_rate':0.01,'loss':'linear'}
    
    AdaBoost_model = AdaBoostRegressor(**adaB_params)
    
    linearModel = LinearRegression()
    
    model.fit(trainX,trainY)
    AdaBoost_model.fit(trainX,trainY)
    linearModel.fit(trainX,trainY)
    
    model_score = model.score(trainX,trainY)
    model_score_test = model.score(testX,testY)
    
    adaBmodel_score = AdaBoost_model.score(trainX,trainY)
    adaBmodel_score_test = AdaBoost_model.score(testX,testY)
    
    linearModel_score = linearModel.score(trainX,trainY)
    linearModel_score_test = linearModel.score(testX,testY)
    
    
    print('R2 sq using Gradient Boosting on Train: ',model_score)
    print('R2 sq using Gradient Boosting on Test: ',model_score_test)
    
    print('\nR2 sq using AdaBoost on Train: ',adaBmodel_score)
    print('R2 sq using AdaBoost on Test: ',adaBmodel_score_test)
    
    print('\nR2 sq using Linear Reg on Train: ',linearModel_score)
    print('R2 sq using Linear Reg on Test: ',linearModel_score_test)
    
    
    predictedY = model.predict(testX)
    predictedY_adaB = AdaBoost_model.predict(testX)
    predictedY_linearModel = linearModel.predict(testX)
    
    # The mean squared error (MSE)
    print("\nMean squared error using Gradient Boosting: %.2f"% mean_squared_error(testY, predictedY))
    print('Test Variance score using Gradient Boosting: %.2f' % r2_score(testY, predictedY))
    
    print("\nMean squared error using AdaBoost: %.2f"% mean_squared_error(testY, predictedY_adaB))
    print('Test Variance score using AdaBoost: %.2f' % r2_score(testY, predictedY_adaB))
    
    print("\nMean squared error using linearReg: %.2f"% mean_squared_error(testY, predictedY_linearModel))
    print('Test Variance score using linearReg %.2f' % r2_score(testY, predictedY_linearModel))
    

    feature_importance = model.feature_importances_
    cls = processed_dt.columns
    feature_imp = {}
    for p,q in zip(cls,feature_importance):
        feature_imp.update({p:q})
    feature_imp_df = pd.DataFrame(feature_imp,columns=cls,index=['importance'])
    feature_imp_df = feature_imp_df.T
    feature_imp_df.sort_values(by='importance',ascending=False,inplace=True)
    return feature_imp_df,predictedY

feature_imp_df,predictedY = run_models(processed_data,y_num)
print (feature_imp_df)

print ("Check the results() function for the model evaluation metrics.....")
print ("performance metrics in results()....")
print ("Test set sizes of 0.10 and 0.30 are taken")

def results():
    
    """
    Gradient Boosting (test size = 0.10):
        R2 sq:  0.8318902181595983
        R2 sq for Test Data:  0.5623284330880953
        Mean squared error: 37383971.55
        Test Variance score: 0.56 with test size of 0.10
    
    Gradient Boosting vs AdaBoost (test size = 0.30):
        
        R2 sq using Gradient Boosting on Train:  0.8258266338084383
        R2 sq using Gradient Boosting on Test:  0.30127694136843197
        R2 sq using AdaBoost on Train:  0.7998400767373118
        R2 sq using AdaBoost on Train:  0.3803268188148172
        Mean squared error using Gradient Boosting: 189864509.40
        Test Variance score using Gradient Boosting: 0.30
        Mean squared error using AdaBoost: 168384230.47
        Test Variance score using AdaBoost: 0.38 on test size of 0.30

    """
    pass

corr_target = pd.DataFrame(corr_target)
corr_target.sort_values(by='loan_amount',ascending=False,inplace=True)


def feature_importance_analysis():
    
    """
    from correlation of numeric variables and feature importance of model,
    The important features observed are:
        1.loan_installments
        2.annual_income
        3.loan_tenure
        4.monthly_expenses
        5.primary_business
        6.loan_purpose
        7.sanitary_availability
        8.social_class
    
    -> This makes sense!!!!!!
    """
    pass


#only numeric cols consider
print ("Considering only Numeric variables for the model....")
numeric_val_imp,predictedY_num = run_models(numeric_df_num,y_num)
print (numeric_val_imp)

#check the significant features among numeric variables

#only factor variables consider
print ("Considering only factor variables for the model....")
char_val_imp,predictedY_char = run_models(char_df_num,y_num)
print (char_val_imp)

#check the significant features among factor variables


print ("Another method of converting categorical variables into numeric....")
print ("****************")

print (""" check the possibility of concatenation/aggregation 
       of levels based on freq scores for all factor columns""")


char_df = data[factor_imp_cols]

def level_cnt(DT,colm,level):
    num = DT[DT[colm]==level].shape[0]
    den = DT[colm].shape[0]
    freq_score = num/den
    return freq_score

def factor_analysis(dt):
    levels_count = {}
    for c in dt.columns:
        levels = pd.Categorical(dt[c]).categories.tolist()
        levels_count[c] = {r:level_cnt(dt,c,r) for r in levels}
    return levels_count

freq_scores = factor_analysis(char_df)

sorted_dict = {}
for s,t in freq_scores.items():
    temp = sorted(t.items(),key=operator.itemgetter(1))
    print (s)
    print (temp)
    temp = {i[0]:i[1] for i in temp}
    print (temp)
    sorted_dict[s] = temp

#Target based encoding
print ("""Mean response value imputation for every category/level""")

#target variable is loan_amount
def mean_response_impute(categs,col,data):
    for cat in categs:
        corr_response = np.mean(data[data[col]==cat]['loan_amount'])
        data[col] = data[col].replace(cat,corr_response)
    return

def target_encoding(cols,data):
    for c in cols:
        cats = pd.Categorical(data[c]).categories
        mean_response_impute(cats,c,data)
    return data
    
dft = pd.concat([char_df,data.iloc[:,-1]],axis=1)

new_char_df = target_encoding(char_df.columns,dft)
new_char_df = new_char_df.iloc[:,:-1]

processed_data_new,y_num_new,numeric_df_num_new,char_df_num_new = prepare_data(new_char_df)
run_models(processed_data_new,y_num_new)


def final_results():
    
    """
    run the function run_models() to see the output
    on console screen
    
    Please find the results below:
        R2 sq using Gradient Boosting on Train:  0.9687830601133741
        R2 sq using Gradient Boosting on Test:  0.7599896683937789
        
        R2 sq using AdaBoost on Train:  0.9198217985855646
        R2 sq using AdaBoost on Test:  0.8034245861387093
        
        R2 sq using Linear Reg on Train:  0.8782295940505308
        R2 sq using Linear Reg on Test:  0.8857894611661451
        
        Mean squared error using Gradient Boosting: 65218176.64
        Test Variance score using Gradient Boosting: 0.76
        
        Mean squared error using AdaBoost: 53415575.82
        Test Variance score using AdaBoost: 0.80
        
        Mean squared error using linearReg: 31034510.25
        Test Variance score using linearReg 0.89
    
                             importance
     social_class             0.706883
     city                     0.130825
     loan_purpose             0.037474
     primary_business         0.030334
     monthly_expenses         0.025130
     age                      0.021731
     secondary_business       0.015008
     house_area               0.014364
     annual_income            0.006101
     loan_tenure              0.004066
     loan_installments        0.003354
     water_availabity         0.002357
     occupants_count          0.001632
     type_of_house            0.000440
     sanitary_availability    0.000191
     young_dependents         0.000070
     old_dependents           0.000040
    
    """
    
    pass

print("""
      Answers:

      1.Among the above three models, GradientBoosting algorithm is
      performing well.
      
      2.Adaptive Boosting Alg is not working well in training set itself.
      
      3.For Linear Regression, The test set error is greater than training
      set error - which is impossible.
      
      4.Yes, loan_purpose is a significant predictor
      
      5.Check the above final_results() function for significant predictors
      or variables.
      
      6.Descriptive analysis/Initial variable screening is done above in the code.
      
      """)

print ("END")

