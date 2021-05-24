#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


# In[2]:


data=pd.read_excel('C:\pbl_movie.xlsx',header=0)


# In[3]:


X=pd.DataFrame()
y=pd.DataFrame()
y1=pd.DataFrame()


# In[147]:


#애니메이션, 전체관람가 제거(다중공선성 문제 해결 위해)
X=data[['주연 top50 출연 여부','배급사','국적','전국 스크린수','경쟁작',
       '가족','공연','공포(호러)','기타','다큐멘터리','드라마','멜로/로맨스','뮤지컬',
       '미스터리','범죄','사극','스릴러','액션','어드벤처','전쟁',
       '코미디','판타지','SF','top영화감독 여부','네티즌 평점','러닝타임',
       '연작','원작','12세관람가','15세관람가','19세관람가',
       '연휴기간 상영여부 (연휴 기간 상영 영화중 점유율 TOP2% (104개정도이고 최저 점유율 대략 17.7%))']]
y=data[['전국 관객수']]
y1=data[['전국 관객수 분류']]


# In[5]:


#단계별 선택법(stepwise selection)으로 변수 선정하기
variables = X.columns.tolist() #설명변수 리스트

selected_variables = [] #선택된 변수들 리스트
sl_enter=0.05 #변수 추가시 유의수준
sl_remove=0.05 #변수 제거시 유의수준

sv_per_step = [] #각 스텝별로 선택된 변수들
adjusted_r_squared = [] #각 스텝별 수정된 결정계수
steps=[] #스텝
step=0
while len(variables) >0:
    remainder = list(set(variables)- set(selected_variables))
    pval=pd.Series(index=remainder) #변수의 p-value리스트
    #기존에 포함된 변수에 새로운 변수 하나씩 돌아가면서 추가해
    #선형 모형에 적합시켜본다
    for col in remainder:
        newX= X[selected_variables+[col]] 
        newX= sm.add_constant(newX)
        model=sm.OLS(y,newX).fit()
        pval[col] = model.pvalues[col] #p-value리스트에 p-value값 넣기
    
    min_pval = pval.min()
    if min_pval < sl_enter: # 최소 p-value값이 유의수준 기준보다 작으면 포함
        selected_variables.append(pval.idxmin())
        #선택된 변수들중에서 어떤 변수를 제거할지 고른다
        while len(selected_variables) >0:
            selected_X= X[selected_variables]
            selected_X = sm.add_constant(selected_X)
            selected_pval= sm.OLS(y,selected_X).fit().pvalues[1:] #절편의 p-value는 뺀다
            max_pval=selected_pval.max()
            if max_pval>=sl_remove: #최대 p-value값이 기준 유의수준보다 크거나 같으면 제외
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
            


# In[6]:


selected_variables


# In[7]:


#단계별 모형 적합도 그래프
fig=plt.figure(figsize=(10,10))
fig.set_facecolor('white')

font_size=15
plt.xticks(steps,[f'step {s}\n'+'\n'.join(sv_per_step[i]) for i,s in enumerate(steps)],fontsize=12)
plt.plot(steps, adjusted_r_squared, marker='o')

plt.ylabel('Adjusted R Squared', fontsize=font_size)
plt.grid(True)
plt.show()


# In[8]:


#데이터들의 상관계수
data.corr()


# In[9]:


# 다중공선성
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif= pd.DataFrame()
vif["VIF Factor"]= [variance_inflation_factor(X.values,i)for i in range(X.shape[1])]
vif["features"] = X.columns
vif


# In[148]:


scaler=StandardScaler()
X=scaler.fit_transform(X)


# In[127]:


#다중선형회귀
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2)
linear=LinearRegression()
start = time.time()
linear.fit(X_train, y_train)
y_pred=linear.predict(X_test)
#예측값이 음수인 경우 0으로 수정
for i in range(0,199):
    if y_pred[i]<0:
        y_pred[i]=0
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
print("학습 데이터 점수 : {}".format(linear.score(X_train,y_train)))
print("평가 데이터 점수 : {}".format(linear.score(X_test,y_test)))
print("rmse : {}".format(rmse))
print("time : {}".format(time.time()-start))


# In[128]:


#k-최근접 이웃 알고리즘(회귀)
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2)
knr= KNeighborsRegressor(n_neighbors=13, weights="distance")
knr.fit(X_train,y_train)
y_pred=knr.predict(X_test)
for i in range(0,199):
    if y_pred[i]<0:
        y_pred[i]=0
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
print("학습 데이터 점수 : {}".format(knr.score(X_train,y_train)))
print("평가 데이터 점수 : {}".format(knr.score(X_test,y_test)))
print("rmse : {}".format(rmse))


# In[129]:


#서포트 벡터 머신(분류)
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


# In[130]:


#크로스탭
y1_test=y1_test.values.ravel()
ct=pd.crosstab(index=y1_test,columns=y_pred)
ct


# In[149]:


#다항선형회귀(2차식)
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1,test_size=0.2)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.fit_transform(X_test)
linear.fit(X_train_poly,y_train)
print("학습 데이터 점수 : {}".format(linear.score(X_train_poly,y_train)))
print("평가 데이터 점수 : {}".format(linear.score(X_test_poly,y_test)))


# In[19]:


#서포트 벡터 머신(회귀) 커널 함수 바꿔가면서 해본 것 linear 말고는 결과 안좋음
X_train,X_test,y_train,y_test= train_test_split(X,y,random_state=1,test_size=0.2)
svr_linear =SVR(kernel='linear', C=260000)
svr_linear.fit(X_train,y_train)
print("학습 데이터 점수 : {}".format(svr_linear.score(X_train,y_train)))
print("평가 데이터 점수 : {}".format(svr_linear.score(X_test,y_test)))
svr_poly=SVR(kernel='poly',C=260000)
svr_poly.fit(X_train,y_train)
print("학습 데이터 점수 : {}".format(svr_poly.score(X_train,y_train)))
print("평가 데이터 점수 : {}".format(svr_poly.score(X_test,y_test)))
svr_rbf=SVR(kernel='rbf',C=260000)
svr_rbf.fit(X_train,y_train)
print("학습 데이터 점수 : {}".format(svr_rbf.score(X_train,y_train)))
print("평가 데이터 점수 : {}".format(svr_rbf.score(X_test,y_test)))


# In[21]:


#서포트 벡터 머신(회귀)
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


# In[22]:


#나이브 베이지안 분류 (다항분포) 이때는 standardscaler적용이 안되서 minmax사용, 각 특성의 평균값을 고려
#각 특성이 어떤 것의 개수를 나타내는 정수값을 가질 때 주로 사용
minmax= MinMaxScaler()
X=minmax.fit_transform(X)
X_train3,X_test3,y1_train,y1_test= train_test_split(X,y1,test_size=0.2)
mn = MultinomialNB(alpha=2,fit_prior=False)


# In[23]:


mn.fit(X_train3,y1_train)
y_pred=mn.predict(X_test3)
print("학습 데이터 점수 : {}".format(mn.score(X_train3,y1_train)))
print("평가 데이터 점수 : {}".format(mn.score(X_test3,y1_test)))


# In[24]:


#크로스탭
y1_test=y1_test.values.ravel()
ct=pd.crosstab(index=y1_test,columns=y_pred)
ct


# In[25]:


#나이브 베이지안 분류(정규분포) , 각 특성의 평균과 표준편차 고려
#연속형 데이터에 주로 사용
minmax= MinMaxScaler()
X=minmax.fit_transform(X)
X_train3,X_test3,y1_train,y1_test= train_test_split(X,y1,test_size=0.2)
gnb= GaussianNB(var_smoothing=0.7)


# In[26]:


gnb.fit(X_train3,y1_train)
y_pred=gnb.predict(X_test3)
print("학습 데이터 점수 : {}".format(gnb.score(X_train3,y1_train)))
print("평가 데이터 점수 : {}".format(gnb.score(X_test3,y1_test)))


# In[27]:


#크로스탭
y1_test=y1_test.values.ravel()
ct=pd.crosstab(index=y1_test,columns=y_pred)
ct


# In[28]:


#다항 로지스틱
X=scaler.fit_transform(X)
X_train3,X_test3,y1_train,y1_test= train_test_split(X,y1,test_size=0.2)
logistic=LogisticRegression(C=0.3,multi_class='multinomial',solver='lbfgs')


# In[29]:


logistic.fit(X_train3,y1_train)
y_pred=logistic.predict(X_test3)
print("학습 데이터 점수 : {}".format(logistic.score(X_train3,y1_train)))
print("평가 데이터 점수 : {}".format(logistic.score(X_test3,y1_test)))


# In[30]:


#크로스탭
y1_test=y1_test.values.ravel()
ct=pd.crosstab(index=y1_test,columns=y_pred)
ct


# In[31]:


#K-최근접 이웃 알고리즘(분류)
X_train3,X_test3,y1_train,y1_test= train_test_split(X,y1,test_size=0.2)
knn=KNeighborsClassifier(n_neighbors=5)


# In[32]:


knn.fit(X_train3,y1_train)
y_pred=knn.predict(X_test3)
print("학습 데이터 점수 : {}".format(knn.score(X_train3,y1_train)))
print("평가 데이터 점수 : {}".format(knn.score(X_test3,y1_test)))


# In[33]:


#크로스탭
y1_test=y1_test.values.ravel()
ct=pd.crosstab(index=y1_test,columns=y_pred)
ct


# In[34]:


#hard voting 앙상블
model1=SVC(kernel='linear', C=0.04)
model2=GaussianNB(var_smoothing=0.7)
model3=RandomForestClassifier()
model4=KNeighborsClassifier(n_neighbors=5)
model5=LogisticRegression(C=0.3,multi_class='multinomial',solver='lbfgs')
vote_model = VotingClassifier(
    estimators=[('svc',model1),('naive',model2),('forest',model3),('knn',model4),('log',model5)],
    voting='hard')
X_train3,X_test3,y1_train,y1_test= train_test_split(X,y1,test_size=0.2)
vote_model.fit(X_train3,y1_train)
print("학습 데이터 점수 : {}".format(vote_model.score(X_train3,y1_train)))
print("평가 데이터 점수 : {}".format(vote_model.score(X_test3,y1_test)))


# In[35]:


#soft voting 앙상블
model1=SVC(kernel='linear', C=0.04, probability=True)
model2=GaussianNB(var_smoothing=0.7)
model3=RandomForestClassifier()
model4=KNeighborsClassifier(n_neighbors=5)
model5=LogisticRegression(C=0.3,multi_class='multinomial',solver='lbfgs')
vote_model = VotingClassifier(
    estimators=[('svc',model1),('naive',model2),('forest',model3),('knn',model4),('log',model5)],
    voting='soft',
    weights=[2,1,2,1,2])
X_train3,X_test3,y1_train,y1_test=train_test_split(X,y1,test_size=0.2)
vote_model.fit(X_train3,y1_train)
print("학습 데이터 점수 : {}".format(vote_model.score(X_train3,y1_train)))
print("평가 데이터 점수 : {}".format(vote_model.score(X_test3,y1_test)))


# In[36]:


#voting 앙상블 회귀
reg1 = LinearRegression()
reg2 = SVR(C=200000)
reg3 = KNeighborsRegressor(n_neighbors=13, weights='distance')
reg4= GradientBoostingRegressor()
reg5= RandomForestRegressor()
vote_reg_model = VotingRegressor(
    estimators=[('linear',reg1),('svr',reg2),('knr',reg3),('gbr',reg4),('rfr',reg5)],
    weights=[1,1,1,1,1])


# In[37]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
vote_reg_model.fit(X_train,y_train)
print("학습 데이터 점수 : {}".format(vote_reg_model.score(X_train,y_train)))
print("평가 데이터 점수 : {}".format(vote_reg_model.score(X_test,y_test)))


# In[143]:


# 상관계수 0.2 이상
X=data[['주연 top50 출연 여부','배급사','전국 스크린수',
       '러닝타임','네티즌 평점','top영화감독 여부',
       '연휴기간 상영여부 (연휴 기간 상영 영화중 점유율 TOP2% (104개정도이고 최저 점유율 대략 17.7%))']]
y=data[['전국 관객수']]
y1=data[['전국 관객수 분류']]
X=scaler.fit_transform(X1)


# In[102]:


#다중선형회귀
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2)
linear=LinearRegression()
start = time.time()
linear.fit(X_train, y_train)
y_pred=linear.predict(X_test)
#예측값이 음수인 경우 0으로 수정
for i in range(0,199):
    if y_pred[i]<0:
        y_pred[i]=0
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
print("학습 데이터 점수 : {}".format(linear.score(X_train,y_train)))
print("평가 데이터 점수 : {}".format(linear.score(X_test,y_test)))
print("rmse : {}".format(rmse))
print("time : {}".format(time.time()-start))


# In[103]:


#k-최근접 이웃 알고리즘(회귀)
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2)
knr= KNeighborsRegressor(n_neighbors=13, weights="distance")
knr.fit(X_train,y_train)
y_pred=knr.predict(X_test)
for i in range(0,199):
    if y_pred[i]<0:
        y_pred[i]=0
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
print("학습 데이터 점수 : {}".format(knr.score(X_train,y_train)))
print("평가 데이터 점수 : {}".format(knr.score(X_test,y_test)))
print("rmse : {}".format(rmse))


# In[104]:


#서포트 벡터 머신(분류)
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


# In[105]:


#크로스탭
y1_test=y1_test.values.ravel()
ct=pd.crosstab(index=y1_test,columns=y_pred)
ct


# In[146]:


#다항선형회귀(2차식)
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1,test_size=0.2)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.fit_transform(X_test)
linear.fit(X_train_poly,y_train)
print("학습 데이터 점수 : {}".format(linear.score(X_train_poly,y_train)))
print("평가 데이터 점수 : {}".format(linear.score(X_test_poly,y_test)))


# In[107]:


#서포트 벡터 머신(회귀) 커널 함수 바꿔가면서 해본 것
X_train,X_test,y_train,y_test= train_test_split(X,y,random_state=1,test_size=0.2)
svr_linear =SVR(kernel='linear',C=260000)
svr_linear.fit(X_train,y_train)
print("학습 데이터 점수 : {}".format(svr_linear.score(X_train,y_train)))
print("평가 데이터 점수 : {}".format(svr_linear.score(X_test,y_test)))
svr_poly=SVR(kernel='poly',C=260000)
svr_poly.fit(X_train,y_train)
print("학습 데이터 점수 : {}".format(svr_poly.score(X_train,y_train)))
print("평가 데이터 점수 : {}".format(svr_poly.score(X_test,y_test)))
svr_rbf=SVR(kernel='rbf',C=260000)
svr_rbf.fit(X_train,y_train)
print("학습 데이터 점수 : {}".format(svr_rbf.score(X_train,y_train)))
print("평가 데이터 점수 : {}".format(svr_rbf.score(X_test,y_test)))


# In[108]:


#서포트 벡터 머신(회귀)
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


# In[109]:


#나이브 베이지안 분류 (다항분포) 이때는 standardscaler적용이 안되서 minmax사용, 각 특성의 평균값을 고려
#각 특성이 어떤 것의 개수를 나타내는 정수값을 가질 때 주로 사용
minmax= MinMaxScaler()
X=minmax.fit_transform(X)
X_train3,X_test3,y1_train,y1_test= train_test_split(X,y1,test_size=0.2)
mn = MultinomialNB(alpha=2,fit_prior=False)


# In[110]:


mn.fit(X_train3,y1_train)
y_pred=mn.predict(X_test3)
print("학습 데이터 점수 : {}".format(mn.score(X_train3,y1_train)))
print("평가 데이터 점수 : {}".format(mn.score(X_test3,y1_test)))


# In[111]:


#크로스탭
y1_test=y1_test.values.ravel()
ct=pd.crosstab(index=y1_test,columns=y_pred)
ct


# In[112]:


#나이브 베이지안 분류(정규분포) , 각 특성의 평균과 표준편차 고려
#연속형 데이터에 주로 사용
minmax= MinMaxScaler()
X=minmax.fit_transform(X)
X_train3,X_test3,y1_train,y1_test= train_test_split(X,y1,test_size=0.2)
gnb= GaussianNB(var_smoothing=0.7)


# In[113]:


gnb.fit(X_train3,y1_train)
y_pred=gnb.predict(X_test3)
print("학습 데이터 점수 : {}".format(gnb.score(X_train3,y1_train)))
print("평가 데이터 점수 : {}".format(gnb.score(X_test3,y1_test)))


# In[114]:


#크로스탭
y1_test=y1_test.values.ravel()
ct=pd.crosstab(index=y1_test,columns=y_pred)
ct


# In[115]:


#다항 로지스틱
X=scaler.fit_transform(X)
X_train3,X_test3,y1_train,y1_test= train_test_split(X,y1,test_size=0.2)
logistic=LogisticRegression(C=0.3,multi_class='multinomial',solver='lbfgs')


# In[116]:


logistic.fit(X_train3,y1_train)
y_pred=logistic.predict(X_test3)
print("학습 데이터 점수 : {}".format(logistic.score(X_train3,y1_train)))
print("평가 데이터 점수 : {}".format(logistic.score(X_test3,y1_test)))


# In[117]:


#크로스탭
y1_test=y1_test.values.ravel()
ct=pd.crosstab(index=y1_test,columns=y_pred)
ct


# In[118]:


#K-최근접 이웃 알고리즘(분류)
X_train3,X_test3,y1_train,y1_test= train_test_split(X,y1,test_size=0.2)
knn=KNeighborsClassifier(n_neighbors=5)


# In[119]:


knn.fit(X_train3,y1_train)
y_pred=knn.predict(X_test3)
print("학습 데이터 점수 : {}".format(knn.score(X_train3,y1_train)))
print("평가 데이터 점수 : {}".format(knn.score(X_test3,y1_test)))


# In[120]:


#크로스탭
y1_test=y1_test.values.ravel()
ct=pd.crosstab(index=y1_test,columns=y_pred)
ct


# In[121]:


#hard voting 앙상블
model1=SVC(kernel='linear', C=0.04)
model2=GaussianNB(var_smoothing=0.7)
model3=RandomForestClassifier()
model4=KNeighborsClassifier(n_neighbors=5)
model5=LogisticRegression(C=0.3,multi_class='multinomial',solver='lbfgs')
vote_model = VotingClassifier(
    estimators=[('svc',model1),('naive',model2),('forest',model3),('knn',model4),('log',model5)],
    voting='hard')
X_train3,X_test3,y1_train,y1_test= train_test_split(X,y1,test_size=0.2)
vote_model.fit(X_train3,y1_train)
print("학습 데이터 점수 : {}".format(vote_model.score(X_train3,y1_train)))
print("평가 데이터 점수 : {}".format(vote_model.score(X_test3,y1_test)))


# In[122]:


#soft voting 앙상블
model1=SVC(kernel='linear', C=0.04, probability=True)
model2=GaussianNB(var_smoothing=0.7)
model3=RandomForestClassifier()
model4=KNeighborsClassifier(n_neighbors=5)
model5=LogisticRegression(C=0.3,multi_class='multinomial',solver='lbfgs')
vote_model = VotingClassifier(
    estimators=[('svc',model1),('naive',model2),('forest',model3),('knn',model4),('log',model5)],
    voting='soft',
    weights=[2,1,2,1,2])
X_train3,X_test3,y1_train,y1_test=train_test_split(X,y1,test_size=0.2)
vote_model.fit(X_train3,y1_train)
print("학습 데이터 점수 : {}".format(vote_model.score(X_train3,y1_train)))
print("평가 데이터 점수 : {}".format(vote_model.score(X_test3,y1_test)))


# In[123]:


#voting 앙상블 회귀
reg1 = LinearRegression()
reg2 = SVR(C=200000)
reg3 = KNeighborsRegressor(n_neighbors=13, weights='distance')
reg4= GradientBoostingRegressor()
reg5= RandomForestRegressor()
vote_reg_model = VotingRegressor(
    estimators=[('linear',reg1),('svr',reg2),('knr',reg3),('gbr',reg4),('rfr',reg5)],
    weights=[1,1,1,1,1])


# In[124]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
vote_reg_model.fit(X_train,y_train)
print("학습 데이터 점수 : {}".format(vote_reg_model.score(X_train,y_train)))
print("평가 데이터 점수 : {}".format(vote_reg_model.score(X_test,y_test)))


# In[133]:


#step-wise
X=data[['주연 top50 출연 여부','전국 스크린수',
        '연휴기간 상영여부 (연휴 기간 상영 영화중 점유율 TOP2% (104개정도이고 최저 점유율 대략 17.7%))',
        '15세관람가','top영화감독 여부','네티즌 평점','국적']]
y=data[['전국 관객수']]
y1=data[['전국 관객수 분류']]
X=scaler.fit_transform(X)


# In[134]:


#다중선형회귀
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2)
linear=LinearRegression()
start = time.time()
linear.fit(X_train, y_train)
y_pred=linear.predict(X_test)
#예측값이 음수인 경우 0으로 수정
for i in range(0,199):
    if y_pred[i]<0:
        y_pred[i]=0
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
print("학습 데이터 점수 : {}".format(linear.score(X_train,y_train)))
print("평가 데이터 점수 : {}".format(linear.score(X_test,y_test)))
print("rmse : {}".format(rmse))
print("time : {}".format(time.time()-start))


# In[135]:


#k-최근접 이웃 알고리즘(회귀)
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2)
knr= KNeighborsRegressor(n_neighbors=13, weights="distance")
knr.fit(X_train,y_train)
y_pred=knr.predict(X_test)
for i in range(0,199):
    if y_pred[i]<0:
        y_pred[i]=0
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
print("학습 데이터 점수 : {}".format(knr.score(X_train,y_train)))
print("평가 데이터 점수 : {}".format(knr.score(X_test,y_test)))
print("rmse : {}".format(rmse))


# In[136]:


#서포트 벡터 머신(분류)
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


# In[137]:


#크로스탭
y1_test=y1_test.values.ravel()
ct=pd.crosstab(index=y1_test,columns=y_pred)
ct


# In[142]:


#다항선형회귀(2차식)
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2)
X_train, X_test,y_train,y_test=train_test_split(X,y,random_state=1,test_size=0.2)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.fit_transform(X_test)
linear.fit(X_train_poly,y_train)
print("학습 데이터 점수 : {}".format(linear.score(X_train_poly,y_train)))
print("평가 데이터 점수 : {}".format(linear.score(X_test_poly,y_test)))


# In[83]:


#서포트 벡터 머신(회귀) 커널 함수 바꿔가면서 해본 것 linear 말고는 결과 안좋음
X_train,X_test,y_train,y_test= train_test_split(X,y,random_state=1,test_size=0.2)
svr_linear =SVR(kernel='linear',C=260000)
svr_linear.fit(X_train,y_train)
print("학습 데이터 점수 : {}".format(svr_linear.score(X_train,y_train)))
print("평가 데이터 점수 : {}".format(svr_linear.score(X_test,y_test)))
svr_poly=SVR(kernel='poly',C=260000)
svr_poly.fit(X_train,y_train)
print("학습 데이터 점수 : {}".format(svr_poly.score(X_train,y_train)))
print("평가 데이터 점수 : {}".format(svr_poly.score(X_test,y_test)))
svr_rbf=SVR(kernel='rbf',C=260000)
svr_rbf.fit(X_train,y_train)
print("학습 데이터 점수 : {}".format(svr_rbf.score(X_train,y_train)))
print("평가 데이터 점수 : {}".format(svr_rbf.score(X_test,y_test)))


# In[84]:


#서포트 벡터 머신(회귀)
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


# In[85]:


#나이브 베이지안 분류 (다항분포) 이때는 standardscaler적용이 안되서 minmax사용, 각 특성의 평균값을 고려
#각 특성이 어떤 것의 개수를 나타내는 정수값을 가질 때 주로 사용
minmax= MinMaxScaler()
X=minmax.fit_transform(X)
X_train3,X_test3,y1_train,y1_test= train_test_split(X,y1,test_size=0.2)
mn = MultinomialNB(alpha=2,fit_prior=False)


# In[86]:


mn.fit(X_train3,y1_train)
y_pred=mn.predict(X_test3)
print("학습 데이터 점수 : {}".format(mn.score(X_train3,y1_train)))
print("평가 데이터 점수 : {}".format(mn.score(X_test3,y1_test)))


# In[87]:


#크로스탭
y1_test=y1_test.values.ravel()
ct=pd.crosstab(index=y1_test,columns=y_pred)
ct


# In[88]:


#나이브 베이지안 분류(정규분포) , 각 특성의 평균과 표준편차 고려
#연속형 데이터에 주로 사용
minmax= MinMaxScaler()
X=minmax.fit_transform(X)
X_train3,X_test3,y1_train,y1_test= train_test_split(X,y1,test_size=0.2)
gnb= GaussianNB(var_smoothing=0.7)


# In[89]:


gnb.fit(X_train3,y1_train)
y_pred=gnb.predict(X_test3)
print("학습 데이터 점수 : {}".format(gnb.score(X_train3,y1_train)))
print("평가 데이터 점수 : {}".format(gnb.score(X_test3,y1_test)))


# In[90]:


#크로스탭
y1_test=y1_test.values.ravel()
ct=pd.crosstab(index=y1_test,columns=y_pred)
ct


# In[91]:


#다항 로지스틱
X=scaler.fit_transform(X)
X_train3,X_test3,y1_train,y1_test= train_test_split(X,y1,test_size=0.2)
logistic=LogisticRegression(C=0.3,multi_class='multinomial',solver='lbfgs')


# In[92]:


logistic.fit(X_train3,y1_train)
y_pred=logistic.predict(X_test3)
print("학습 데이터 점수 : {}".format(logistic.score(X_train3,y1_train)))
print("평가 데이터 점수 : {}".format(logistic.score(X_test3,y1_test)))


# In[93]:


#크로스탭
y1_test=y1_test.values.ravel()
ct=pd.crosstab(index=y1_test,columns=y_pred)
ct


# In[94]:


#K-최근접 이웃 알고리즘(분류)
X_train3,X_test3,y1_train,y1_test= train_test_split(X,y1,test_size=0.2)
knn=KNeighborsClassifier(n_neighbors=5)


# In[95]:


knn.fit(X_train3,y1_train)
y_pred=knn.predict(X_test3)
print("학습 데이터 점수 : {}".format(knn.score(X_train3,y1_train)))
print("평가 데이터 점수 : {}".format(knn.score(X_test3,y1_test)))


# In[96]:


#크로스탭
y1_test=y1_test.values.ravel()
ct=pd.crosstab(index=y1_test,columns=y_pred)
ct


# In[97]:


#hard voting 앙상블
model1=SVC(kernel='linear', C=0.04)
model2=GaussianNB(var_smoothing=0.7)
model3=RandomForestClassifier()
model4=KNeighborsClassifier(n_neighbors=5)
model5=LogisticRegression(C=0.3,multi_class='multinomial',solver='lbfgs')
vote_model = VotingClassifier(
    estimators=[('svc',model1),('naive',model2),('forest',model3),('knn',model4),('log',model5)],
    voting='hard')
X_train3,X_test3,y1_train,y1_test= train_test_split(X,y1,test_size=0.2)
vote_model.fit(X_train3,y1_train)
print("학습 데이터 점수 : {}".format(vote_model.score(X_train3,y1_train)))
print("평가 데이터 점수 : {}".format(vote_model.score(X_test3,y1_test)))


# In[98]:


#soft voting 앙상블
model1=SVC(kernel='linear', C=0.04, probability=True)
model2=GaussianNB(var_smoothing=0.7)
model3=RandomForestClassifier()
model4=KNeighborsClassifier(n_neighbors=5)
model5=LogisticRegression(C=0.3,multi_class='multinomial',solver='lbfgs')
vote_model = VotingClassifier(
    estimators=[('svc',model1),('naive',model2),('forest',model3),('knn',model4),('log',model5)],
    voting='soft',
    weights=[2,1,2,1,2])
X_train3,X_test3,y1_train,y1_test=train_test_split(X,y1,test_size=0.2)
vote_model.fit(X_train3,y1_train)
print("학습 데이터 점수 : {}".format(vote_model.score(X_train3,y1_train)))
print("평가 데이터 점수 : {}".format(vote_model.score(X_test3,y1_test)))


# In[99]:


#voting 앙상블 회귀
reg1 = LinearRegression()
reg2 = SVR(C=200000)
reg3 = KNeighborsRegressor(n_neighbors=13, weights='distance')
reg4= GradientBoostingRegressor()
reg5= RandomForestRegressor()
vote_reg_model = VotingRegressor(
    estimators=[('linear',reg1),('svr',reg2),('knr',reg3),('gbr',reg4),('rfr',reg5)],
    weights=[1,1,1,1,1])


# In[100]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
vote_reg_model.fit(X_train,y_train)
print("학습 데이터 점수 : {}".format(vote_reg_model.score(X_train,y_train)))
print("평가 데이터 점수 : {}".format(vote_reg_model.score(X_test,y_test)))


# In[172]:


#코로나 여파, 감독의 이전작품 평균 관객수 데이터 넣은것
data=pd.read_excel('C:\\Users\wjdtj\\Desktop\\pbl_movie2.xlsx',header=0)


# In[173]:


X=pd.DataFrame()
y=pd.DataFrame()
y1=pd.DataFrame()


# In[174]:


data.corr()


# In[175]:


#애니메이션, 전체관람가 제거
X=data[['주연 top50 출연 여부','배급사','국적','전국 스크린수','경쟁작',
       '가족','공연','공포(호러)','기타','다큐멘터리','드라마','멜로/로맨스','뮤지컬',
       '미스터리','범죄','사극','스릴러','액션','어드벤처','전쟁',
       '코미디','판타지','SF','2005년 이후 감독의 해당 영화 개봉전 평균 관객수(20만 이상만)','네티즌 평점','러닝타임',
       '연작','원작','12세관람가','15세관람가','19세관람가','코로나 여파',
       '연휴기간 상영여부 (연휴 기간 상영 영화중 점유율 TOP2% (104개정도이고 최저 점유율 대략 17.7%))']]
y=data[['전국 관객수']]
y1=data[['전국 관객수 분류']]


# In[176]:


scaler=StandardScaler()
X=scaler.fit_transform(X)


# In[177]:


X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2)
linear=LinearRegression()


# In[178]:


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


# In[179]:


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

