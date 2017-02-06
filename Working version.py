import pandas as pd
import matplotlib as plt
import numpy as np
import pandas_datareader.data as web
from matplotlib import style
import matplotlib.pyplot as plt 
style.use('ggplot')

df = pd.read_csv('UCI_Credit_Card.csv', index_col=0)

##format varibles
df['SEX'] = np.where(df['SEX'] == 1, 'M', 'F')
df['event'] = np.where(df['default.payment.next.month'] == 1, 1, 0)
df['non-event'] = np.where(df['default.payment.next.month'] == 0, 1, 0)
df = df.rename(columns={'LIMIT_BAL':'credit limit'})

col_names = df.columns.values.tolist()
##print('colnames')
##print(len(col_names))
##print(col_names)


def make_woe_set(dset,var,inc_error=0):
    if inc_error==0:
        var_summ = dset.groupby(var).agg({'event':np.sum, 'non-event':np.sum})
        var_summ['total'] = var_summ['event'] + var_summ['non-event']
        var_summ['total_events'] = dset['event'].sum()
        var_summ['total_non-events'] = dset['non-event'].sum()
        var_summ['% events'] = var_summ['event'] / var_summ['total_events']
        var_summ['% non-events'] = var_summ['non-event'] / var_summ['total_non-events']
        var_summ['weight of evidence'] = np.log(var_summ['% non-events']/var_summ['% events'])
        var_summ['infomation value'] = (var_summ['% events'] - var_summ['% non-events']) * var_summ['weight of evidence']
    if inc_error==1:
        var_summ = dset.groupby(var).agg({'event':np.sum, 'non-event':np.sum})
        var_summ['total'] = var_summ['event'] + var_summ['non-event']
        var_summ['total_events'] = dset['event'].sum()
        var_summ['total_non-events'] = dset['non-event'].sum()
        var_summ['% events'] = var_summ['event'] / var_summ['total_events']
        var_summ['% non-events'] = var_summ['non-event'] / var_summ['total_non-events']
        var_summ['weight of evidence'] = np.log(((var_summ['non-event'] + 0.5) / (var_summ['total_non-events'] + 0.5))/((var_summ['event'] + 0.5) / (var_summ['total_events'] + 0.5)))
        var_summ['WoE Variance'] = (1/var_summ['non-event']) + (1/var_summ['event']) - (1/var_summ['total_non-events']) - (1/var_summ['total_events'])
        var_summ['infomation value'] = (var_summ['% events']-var_summ['% non-events'])*var_summ['weight of evidence']                                

    return var_summ

info_val=[]

i=0
while i < len(col_names):  
##    print('step '+str(i))
##    print( len( df[col_names[i]].value_counts() ) )
    if len( df[col_names[i]].value_counts() ) > 100:
        var = str(col_names[i])
        class_var = str(col_names[i]) + '_class'
##        print(class_var)
        df[class_var] = pd.cut(df[var], bins=100, labels=False)
        temp = make_woe_set(df,class_var,1)
        info_val.append(temp['infomation value'].sum())
    else:
        temp = make_woe_set(df,str(col_names[i]),1)
##        print(temp['infomation value'])
        info_val.append(temp['infomation value'].sum())
        
    i+=1

##print(col_names)
##print(info_val)

information_values = pd.DataFrame({'Varible' : col_names,
                                   'Information Value' : info_val
                                  }, columns=['Varible','Information Value'])

information_values.set_index('Varible')

information_values.sort_values('Information Value').to_csv('IV Table.csv', columns=['Varible','Information Value'])

# Logistic Regression
from sklearn.linear_model import LogisticRegression
# fit a logistic regression model to the data
model = LogisticRegression()
model.fit(part['bill_class'],part['default.payment.next.month'])
print(model)
# make predictions
expected = part['default.payment.next.month']
predicted = model.predict(part['bill_class'])
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
    
