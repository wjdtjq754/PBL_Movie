#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR,SVC
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.model_selection import cross_validate, GridSearchCV
import multiprocessing
from sklearn.metrics import mean_squared_error
from scipy import stats
import statsmodels.api as sm
import time
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import VotingRegressor


# In[67]:


data=pd.read_excel('C:\pbl_movie.xlsx',header=0)


# In[68]:


X=pd.DataFrame()
y=pd.DataFrame()
y1=pd.DataFrame()


# In[69]:


#애니메이션, 전체관람가 제거
X=data[['주연 top50 출연 여부','배급사','국적','전국 스크린수','경쟁작',
       '가족','공연','공포(호러)','기타','다큐멘터리','드라마','멜로/로맨스','뮤지컬',
       '미스터리','범죄','사극','스릴러','액션','어드벤처','전쟁',
       '코미디','판타지','SF','top영화감독 여부','네티즌 평점','러닝타임',
       '연작','원작','12세관람가','15세관람가','19세관람가',
       '연휴기간 상영여부 (연휴 기간 상영 영화중 점유율 TOP2% (104개정도이고 최저 점유율 대략 17.7%))']]
y=data[['전국 관객수']]
y1=data[['전국 관객수 분류']]


# In[73]:


variables = X.columns.tolist()

selected_variables = []
sl_enter=0.05
sl_remove=0.05

sv_per_step = []
adjusted_r_squared = []
steps=[]
step=0
while len(variables) >0:
    remainder = list(set(variables)- set(selected_variables))
    pval=pd.Series(index=remainder)
    
    for col in remainder:
        newX= X[selected_variables+[col]] 
        newX= sm.add_constant(newX)
        model=sm.OLS(y,newX).fit()
        pval[col] = model.pvalues[col]
    
    min_pval = pval.min()
    if min_pval < sl_enter:
        selected_variables.append(pval.idxmin())
        
        while len(selected_variables) >0:
            selected_X= X[selected_variables]
            selected_X = sm.add_constant(selected_X)
            selected_pval= sm.OLS(y,selected_X).fit().pvalues[1:]
            max_pval=selected_pval.max()
            if max_pval>=sl_remove:
                remove_variable= selected_pval.idxmax()
                selected_variables.remove(remove_variable)
            else:
                break
        
        step+=1
        steps.append(step)
        adj_r_squared= sm.OLS(y,sm.add_constant(X[selected_variables])).fit().rsquared_adj
        adjusted_r_squared.append(adj_r_squared)
        sv_per_step.append(selected_variables.copy())
    else:
        break
            


# In[74]:


selected_variables


# In[7]:


data.corr()


# In[50]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
vif= pd.DataFrame()
vif["VIF Factor"]= [variance_inflation_factor(X.values,i)for i in range(X.shape[1])]
vif["features"] = X.columns
vif


# In[44]:


scaler=StandardScaler()
X=scaler.fit_transform(X)


# In[63]:


X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2)
linear=LinearRegression()


# In[64]:


start = time.time()
linear.fit(X_train, y_train)
y_pred=linear.predict(X_test)
for i in range(0,199):
    if y_pred[i]<0:
        y_pred[i]=0
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
print("학습 데이터 점수 : {}".format(linear.score(X_train,y_train)))
print("평가 데이터 점수 : {}".format(linear.score(X_test,y_test)))
print("rmse : {}".format(rmse))
print(time.time()-start)


# In[54]:


X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2)
knr= KNeighborsRegressor(n_neighbors=13, weights="distance")


# In[55]:


knr.fit(X_train,y_train)
y_pred=knr.predict(X_test)
for i in range(0,199):
    if y_pred[i]<0:
        y_pred[i]=0
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
print("학습 데이터 점수 : {}".format(knr.score(X_train,y_train)))
print("평가 데이터 점수 : {}".format(knr.score(X_test,y_test)))
print("rmse : {}".format(rmse))


# In[38]:


X_train3,X_test3,y1_train,y1_test= train_test_split(X,y1,test_size=0.2)
svc=SVC(kernel='linear',C=0.04)
start = time.time()
svc.fit(X_train3,y1_train)
y_pred=svc.predict(X_test3)
rmse=np.sqrt(mean_squared_error(y1_test,y_pred))
print("학습 데이터 점수 : {}".format(svc.score(X_train3,y1_train)))
print("평가 데이터 점수 : {}".format(svc.score(X_test3,y1_test)))
print("rmse : {}".format(rmse))
print(time.time()-start)


# In[194]:



y1_test=y1_test.values.ravel()
t=np.c_[y1_test,y_pred]
dt=pd.DataFrame(t)
ct=pd.crosstab(index=y1_test,columns=y_pred)
ct


# In[11]:


from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.fit_transform(X_test)
linear.fit(X_train_poly,y_train)
print("학습 데이터 점수 : {}".format(linear.score(X_train_poly,y_train)))
print("평가 데이터 점수 : {}".format(linear.score(X_test_poly,y_test)))


# In[55]:


X_train,X_test,y_train,y_test= train_test_split(X,y,random_state=1,test_size=0.2)
svr_linear =SVR(kernel='linear')
svr_linear.fit(X_train,y_train)
print("학습 데이터 점수 : {}".format(svr_linear.score(X_train,y_train)))
print("평가 데이터 점수 : {}".format(svr_linear.score(X_test,y_test)))
svr_poly=SVR(kernel='poly')
svr_poly.fit(X_train,y_train)
print("학습 데이터 점수 : {}".format(svr_poly.score(X_train,y_train)))
print("평가 데이터 점수 : {}".format(svr_poly.score(X_test,y_test)))
svr_rbf=SVR(kernel='rbf')
svr_rbf.fit(X_train,y_train)
print("학습 데이터 점수 : {}".format(svr_rbf.score(X_train,y_train)))
print("평가 데이터 점수 : {}".format(svr_rbf.score(X_test,y_test)))


# In[49]:


X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2)

svr_linear2=SVR(kernel='linear',C=260000)
start= time.time()
svr_linear2.fit(X_train,y_train)
y_pred=svr_linear2.predict(X_test)
for i in range(0,199):
    if y_pred[i]<0:
        y_pred[i]=0
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
print("학습 데이터 점수 : {}".format(svr_linear2.score(X_train,y_train)))
print("평가 데이터 점수 : {}".format(svr_linear2.score(X_test,y_test)))
print("rmse : {}".format(rmse))
print(time.time()-start)


# In[55]:


minmax= MinMaxScaler()
X=minmax.fit_transform(X)
X_train3,X_test3,y1_train,y1_test= train_test_split(X,y1,random_state=10,test_size=0.2)
mn = MultinomialNB(alpha=2,fit_prior=False)


# In[56]:


mn.fit(X_train3,y1_train)
y_pred=mn.predict(X_test3)
print("학습 데이터 점수 : {}".format(mn.score(X_train3,y1_train)))
print("평가 데이터 점수 : {}".format(mn.score(X_test3,y1_test)))


# In[111]:


minmax= MinMaxScaler()
X=minmax.fit_transform(X)
X_train3,X_test3,y1_train,y1_test= train_test_split(X,y1,random_state=10,test_size=0.2)
gnb= GaussianNB(var_smoothing=0.7)


# In[112]:


gnb.fit(X_train3,y1_train)
y_pred=gnb.predict(X_test3)
print("학습 데이터 점수 : {}".format(gnb.score(X_train3,y1_train)))
print("평가 데이터 점수 : {}".format(gnb.score(X_test3,y1_test)))


# In[60]:


X=scaler.fit_transform(X)
X_train3,X_test3,y1_train,y1_test= train_test_split(X,y1,test_size=0.2)
logistic=LogisticRegression(C=0.3,multi_class='multinomial',solver='lbfgs')


# In[61]:


logistic.fit(X_train3,y1_train)
y_pred=logistic.predict(X_test3)
print("학습 데이터 점수 : {}".format(logistic.score(X_train3,y1_train)))
print("평가 데이터 점수 : {}".format(logistic.score(X_test3,y1_test)))


# In[78]:


X_train3,X_test3,y1_train,y1_test= train_test_split(X,y1,test_size=0.2)
knn=KNeighborsClassifier(n_neighbors=5)


# In[79]:


knn.fit(X_train3,y1_train)
y_pred=knn.predict(X_test3)
print("학습 데이터 점수 : {}".format(knn.score(X_train3,y1_train)))
print("평가 데이터 점수 : {}".format(knn.score(X_test3,y1_test)))


# In[16]:


model1=SVC(kernel='linear', C=0.04)
model2=GaussianNB(var_smoothing=0.7)
model3=RandomForestClassifier()
model4=KNeighborsClassifier(n_neighbors=5)
model5=LogisticRegression(C=0.3,multi_class='multinomial',solver='lbfgs')
vote_model = VotingClassifier(
    estimators=[('svc',model1),('naive',model2),('forest',model3),('knn',model4),('log',model5)],
    voting='hard')


# In[19]:


for model in (model1,model2,model3,model4,model5, vote_model):
    model_name= str(type(model)).split('.')[-1][:-2]
    scores= cross_val_score(model,X,y1,cv=5)
    print('Accuracy:%0.2f (+/-%0.2f)[%s]' % (scores.mean(), scores.std(), model_name))


# In[20]:


model1=SVC(kernel='linear', C=0.04, probability=True)
model2=GaussianNB(var_smoothing=0.7)
model3=RandomForestClassifier()
model4=KNeighborsClassifier(n_neighbors=5)
model5=LogisticRegression(C=0.3,multi_class='multinomial',solver='lbfgs')
vote_model = VotingClassifier(
    estimators=[('svc',model1),('naive',model2),('forest',model3),('knn',model4),('log',model5)],
    voting='soft',
    weights=[2,1,2,1,2])


# In[22]:


for model in (model1,model2,model3,model4,model5, vote_model):
    model_name= str(type(model)).split('.')[-1][:-2]
    scores= cross_val_score(model,X,y1,cv=5)
    print('Accuracy:%0.2f (+/-%0.2f)[%s]' % (scores.mean(), scores.std(), model_name))


# In[30]:


reg1 = LinearRegression()
reg2 = SVR(C=200000)
reg3 = KNeighborsRegressor(n_neighbors=13, weights='distance')
reg4= GradientBoostingRegressor()
reg5= RandomForestRegressor()
vote_reg_model = VotingRegressor(
    estimators=[('linear',reg1),('svr',reg2),('knr',reg3),('gbr',reg4),('rfr',reg5)],
    weights=[1,1,1,1,1])


# In[35]:


X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=10,test_size=0.2)
vote_reg_model.fit(X_train,y_train)
print(vote_reg_model.score(X_train,y_train))
print(vote_reg_model.score(X_test,y_test))


# In[32]:


for reg in (reg1,reg2,reg3,reg4,reg5,vote_reg_model):
    model_name= str(type(reg)).split('.')[-1][:-2]
    scores = cross_val_score(reg,X,y,cv=5)
    print('R2: %0.2f (+/- %0.2f) [%s]' % (scores.mean(),scores.std(), model_name))


# In[51]:


# 상관계수 0.2 이상
X1=data[['주연 top50 출연 여부','배급사','전국 스크린수',
       '러닝타임','네티즌 평점','top영화감독 여부',
       '연휴기간 상영여부 (연휴 기간 상영 영화중 점유율 TOP2% (104개정도이고 최저 점유율 대략 17.7%))']]
y_1=data[['전국 관객수']]
y1_1=data[['전국 관객수 분류']]
X1=scaler.fit_transform(X1)


# In[61]:


X_train2,X_test2,y_train2,y_test2=train_test_split(X1,y_1 ,test_size=0.2)
svr_linear2=SVR(kernel='linear',C=200000)
start=time.time()
svr_linear2.fit(X_train2,y_train2)
y_pred=svr_linear2.predict(X_test2)
for i in range(0,199):
    if y_pred[i]<0:
        y_pred[i]=0
rmse=np.sqrt(mean_squared_error(y_test2,y_pred))
print("학습 데이터 점수 : {}".format(svr_linear2.score(X_train2,y_train2)))
print("평가 데이터 점수 : {}".format(svr_linear2.score(X_test2,y_test2)))
print("rmse : {}".format(rmse))
print(time.time()-start)


# In[73]:


X_train2,X_test2,y_train2,y_test2=train_test_split(X1,y_1, test_size=0.2)
start=time.time()
linear.fit(X_train2,y_train2)
y_pred=linear.predict(X_test2)
for i in range(0,199):
    if y_pred[i]<0:
        y_pred[i]=0
rmse=np.sqrt(mean_squared_error(y_test2,y_pred))

print("학습 데이터 점수 : {}".format(linear.score(X_train2,y_train2)))
print("평가 데이터 점수 : {}".format(linear.score(X_test2,y_test2)))
print("rmse : {}".format(rmse))
print(time.time()-start)


# In[83]:


X_train4,X_test4,y_1train2,y_1test2=train_test_split(X1,y1_1, test_size=0.2)
svc=SVC(kernel='linear',C=0.04)
start=time.time()
svc.fit(X_train4,y_1train2)
y_pred=svc.predict(X_test4)
rmse=np.sqrt(mean_squared_error(y_1test2,y_pred))
print("학습 데이터 점수 : {}".format(svc.score(X_train4,y_1train2)))
print("평가 데이터 점수 : {}".format(svc.score(X_test4,y_1test2)))
print("rmse : {}".format(rmse))
print(time.time()-start)


# In[268]:



y_1test2=y_1test2.values.ravel()
t=np.c_[y_1test2,y_pred]
dt=pd.DataFrame(t)
ct=pd.crosstab(index=y_1test2,columns=y_pred)
ct


# In[75]:


#step-wise
X=data[['주연 top50 출연 여부','전국 스크린수',
        '연휴기간 상영여부 (연휴 기간 상영 영화중 점유율 TOP2% (104개정도이고 최저 점유율 대략 17.7%))',
        '15세관람가','top영화감독 여부','네티즌 평점','국적']]
y=data[['전국 관객수']]
y1=data[['전국 관객수 분류']]
X=scaler.fit_transform(X)


# In[76]:


X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2)
linear=LinearRegression()


# In[77]:


start=time.time()
linear.fit(X_train, y_train)
y_pred=linear.predict(X_test)
for i in range(0,199):
    if y_pred[i]<0:
        y_pred[i]=0
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
print("학습 데이터 점수 : {}".format(linear.score(X_train,y_train)))
print("평가 데이터 점수 : {}".format(linear.score(X_test,y_test)))
print("rmse : {}".format(rmse))
print(time.time()-start)


# In[78]:


X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2)


# In[79]:


svr_linear2=SVR(kernel='linear',C=200000)
start=time.time()
svr_linear2.fit(X_train,y_train)
y_pred=svr_linear2.predict(X_test)
for i in range(0,199):
    if y_pred[i]<0:
        y_pred[i]=0
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
print("학습 데이터 점수 : {}".format(svr_linear2.score(X_train,y_train)))
print("평가 데이터 점수 : {}".format(svr_linear2.score(X_test,y_test)))
print("rmse : {}".format(rmse))
print(time.time()-start)


# In[80]:


X_train4,X_test4,y_1train2,y_1test2=train_test_split(X,y1,test_size=0.2)
svc=SVC(kernel='linear',C=0.04)
start=time.time()
svc.fit(X_train4,y_1train2)
y_pred=svc.predict(X_test4)
rmse=np.sqrt(mean_squared_error(y_1test2,y_pred))
print("학습 데이터 점수 : {}".format(svc.score(X_train4,y_1train2)))
print("평가 데이터 점수 : {}".format(svc.score(X_test4,y_1test2)))
print("rmse : {}".format(rmse))
print(time.time()-start)


# In[81]:



y_1test2=y_1test2.values.ravel()
t=np.c_[y_1test2,y_pred]
dt=pd.DataFrame(t)
ct=pd.crosstab(index=y_1test2,columns=y_pred)
ct


# In[2]:


data=pd.read_excel('C:\\Users\wjdtj\\Desktop\\pbl_movie2.xlsx',header=0)


# In[3]:


X=pd.DataFrame()
y=pd.DataFrame()
y1=pd.DataFrame()


# In[4]:


data.corr()


# In[5]:


#애니메이션, 전체관람가 제거
X=data[['주연 top50 출연 여부','배급사','국적','전국 스크린수','경쟁작',
       '가족','공연','공포(호러)','기타','다큐멘터리','드라마','멜로/로맨스','뮤지컬',
       '미스터리','범죄','사극','스릴러','액션','어드벤처','전쟁',
       '코미디','판타지','SF','2005년 이후 감독의 해당 영화 개봉전 평균 관객수(20만 이상만)','네티즌 평점','러닝타임',
       '연작','원작','12세관람가','15세관람가','19세관람가','코로나 여파',
       '연휴기간 상영여부 (연휴 기간 상영 영화중 점유율 TOP2% (104개정도이고 최저 점유율 대략 17.7%))']]
y=data[['전국 관객수']]
y1=data[['전국 관객수 분류']]


# In[6]:


scaler=StandardScaler()
X=scaler.fit_transform(X)


# In[82]:


X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2)
linear=LinearRegression()


# In[83]:


start = time.time()
linear.fit(X_train, y_train)
y_pred=linear.predict(X_test)
for i in range(0,199):
    if y_pred[i]<0:
        y_pred[i]=0
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
print("학습 데이터 점수 : {}".format(linear.score(X_train,y_train)))
print("평가 데이터 점수 : {}".format(linear.score(X_test,y_test)))
print("rmse : {}".format(rmse))
print(time.time()-start)


# In[32]:


X_train3,X_test3,y1_train,y1_test= train_test_split(X,y1,test_size=0.2)
svc=SVC(kernel='linear',C=0.04)
start = time.time()
svc.fit(X_train3,y1_train)
y_pred=svc.predict(X_test3)
rmse=np.sqrt(mean_squared_error(y1_test,y_pred))
print("학습 데이터 점수 : {}".format(svc.score(X_train3,y1_train)))
print("평가 데이터 점수 : {}".format(svc.score(X_test3,y1_test)))
print("rmse : {}".format(rmse))
print(time.time()-start)


# In[ ]:




