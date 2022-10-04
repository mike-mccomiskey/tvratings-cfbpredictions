import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor
# from lightgbm.sklearn import LGBMRegressor
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
import pickle

pd.set_option('display.max_rows',1500)
pd.set_option('display.max_columns',100)
df = pd.read_excel('ratings-final1.xlsx')

df['HomeGames'] = df['HomeWins'] + df['HomeLosses']
df['AwayGames'] = df['AwayWins'] + df['AwayLosses']
df['RecDifferentials'] = (df['HomeWins'] - df['HomeLosses']**2) + (df['AwayWins'] - df['AwayLosses']**2)

df['OULine'] = df['OULine'].fillna(df['OULine'].mean())
df['Line'] = df['Line'].fillna(df['Line'].mean())

df2 = df.dropna(subset=['Viewers','HomeOff','HomeDef','AwayOff','AwayDef'])

df3 = df2[(df2['Postseason'] == 0) & (df2['Network'].isin(['ABC','FOX','NBC','CBS','ESPN','ESPN2','FS1']))].reset_index(drop=True)

df3[['TV_OTA','TV_ESPN','TV_Cable_ESPN2','TV_Cable_FS1']] = [0,0,0,0]

def network_grouping(r):
  if (r['Network'] in ['ABC','CBS','NBC','FOX']):
    r['TV_OTA'] = 1
  elif (r['Network'] in ['ESPN']):
    r['TV_ESPN'] = 1
  elif (r['Network'] in ['ESPN2']):
    r['TV_Cable_ESPN2'] = 1
  elif (r['Network'] in ['FS1']):
    r['TV_Cable_FS1'] = 1
  return r

df3 = df3.apply(network_grouping, axis=1)

df3['ViewersK'] = df3['Viewers'] / 1000

df3['HomeOff'] = df3['HomeOff'].replace(0,df3['HomeOff'].mean())
df3['HomeDef'] = df3['HomeDef'].replace(0,df3['HomeDef'].mean())
df3['AwayOff'] = df3['AwayOff'].replace(0,df3['HomeOff'].mean())
df3['AwayDef'] = df3['AwayDef'].replace(0,df3['AwayDef'].mean())

allTeams = pd.concat([df3['Home'],df3['Away']],axis=0,ignore_index=True)
uniqueTeams = np.sort(allTeams.unique())
allYears = [2013,2014,2015,2016,2017,2018,2019,2020,2021,2022]

tempDf = pd.DataFrame(uniqueTeams,columns=['Teams'])
fullList = [(i,j)
             for i in tempDf['Teams']
             for j in allYears]

# print(fullList)

lookupDf = pd.DataFrame(fullList,columns=['Teams','Year'])
lookupDf['TotalViewers'] = 0
lookupDf['TotalCount'] = 0
lookupDf['MeanViewers'] = 0

tempdf1 = pd.read_excel('ratings-final1.xlsx')
tempdf2 = pd.read_excel('ratings-2021.xlsx')
fulldf = pd.concat([tempdf1,tempdf2])
fulldf = fulldf.dropna(subset=['Viewers','HomeOff','HomeDef','AwayOff','AwayDef'])
fulldf = fulldf[(fulldf['Postseason'] == 0) & (fulldf['Network'].isin(['ABC','FOX','NBC','CBS','ESPN','ESPN2','FS1']))].reset_index(drop=True)
def summary_row(r):
  if (r.Year in [2013,2014,2015,2016]):
    yearlist = [2013,2014,2015]
  elif (r.Year == 2017):
    yearlist = [2014,2015,2016]
  elif (r.Year == 2018):
    yearlist = [2015,2016,2017]
  elif (r.Year == 2019):
    yearlist = [2016,2017,2018]
  elif (r.Year == 2020):
    yearlist = [2017,2018,2019]
  elif (r.Year == 2021):
    yearlist = [2018,2019,2020]
  elif (r.Year == 2022):
    yearlist = [2019,2020,2021]
  r.TotalViewers = int(fulldf[(fulldf['Home'] == r.Teams) & (fulldf['Year'].isin(yearlist))]['Viewers'].sum() + fulldf[(fulldf['Away'] == r.Teams) & (fulldf['Year'].isin(yearlist))]['Viewers'].sum())
  r.TotalCount = fulldf[(fulldf['Home'] == r.Teams) & (fulldf['Year'].isin(yearlist))]['Viewers'].count() + fulldf[(fulldf['Away'] == r.Teams) & (fulldf['Year'].isin(yearlist))]['Viewers'].count()
  if r.TotalCount > 0:
    r.MeanViewers = int(r.TotalViewers / r.TotalCount)
  else:
    r.MeanViewers = 0
  return r

lookupDf = lookupDf.apply(summary_row,axis=1)
for i in allYears:
  min_value = lookupDf[lookupDf['Year'] == i]['MeanViewers'].min()
  max_value = lookupDf[lookupDf['Year'] == i]['MeanViewers'].max()
  bins = np.linspace(min_value,max_value,6)
  labels = [1,2,3,4,5]
  lookupDf.loc[lookupDf['Year'] == i,'bins'] = pd.cut(lookupDf[lookupDf['Year'] == i]['MeanViewers'],bins=bins,labels=labels,include_lowest=True)
lookupDf['bins'].fillna(1)
lookupDf.loc[(lookupDf.TotalCount<10)&(lookupDf.bins>3),'bins'] = (lookupDf.bins).astype('int')-1
print(lookupDf['bins'].value_counts())
lookupDf['bins'] = lookupDf['bins'].astype('int64')


# lookupDf['bins'] = lookupDf['bins'].astype('int')

lookupDf.to_csv('lookupdf.csv')

def add_bins(r):
  r.HomeBin = lookupDf[lookupDf['Teams']==r.Home]['bins'].sum()
  r.AwayBin = lookupDf[lookupDf['Teams']==r.Away]['bins'].sum()
  return r

df3['HomeBin'] = 0
df3['AwayBin'] = 0
df3 = df3.apply(add_bins,axis=1)

df3['AddBin'] = df3['HomeBin'] + df3['AwayBin']
df3['MultBin'] = df3['HomeBin'] * df3['AwayBin']

df3['HasACC'] = np.where((df3['HomeConf']=='ACC') | (df3['AwayConf']=='ACC'),1,0)
df3['HasBigTen'] = np.where((df3['HomeConf']=='Big Ten') | (df3['AwayConf']=='Big Ten'),1,0)
df3['HasBig12'] = np.where((df3['HomeConf']=='Big 12') | (df3['AwayConf']=='Big 12'),1,0)
df3['HasPac12'] = np.where((df3['HomeConf']=='Pac-12') | (df3['AwayConf']=='Pac-12'),1,0)
df3['HasSEC'] = np.where((df3['HomeConf']=='SEC') | (df3['AwayConf']=='SEC'),1,0)
df3['HasFCS'] = np.where((df3['HomeConf']=='FCS') | (df3['AwayConf']=='FCS'),1,0)
df3['A5vA5'] = np.where(((df3['HasACC']+df3['HasBigTen']+df3['HasBig12']+df3['HasPac12']+df3['HasSEC']+df3['FBSConfGame'])==2),1,0)
df3['A5vG5'] = np.where(((df3['HasACC']+df3['HasBigTen']+df3['HasBig12']+df3['HasPac12']+df3['HasSEC'])==1)&((df3['FBSConfGame']+df3['HasFCS'])==0),1,0)
df3['G5vG5'] = np.where(((df3['HasACC']+df3['HasBigTen']+df3['HasBig12']+df3['HasPac12']+df3['HasSEC'])==0)&(df3['HasFCS']==0),1,0)

df3['Top2'] = np.where((df3['Home Rank']<=2.0),1,0)+np.where((df3['Away Rank']<=2.0),1,0)
df3['Top5'] = np.where((df3['Home Rank']<=5.0),1,0)+np.where((df3['Away Rank']<=5.0),1,0)
df3['Top10'] = np.where((df3['Home Rank']<=10.0),1,0)+np.where((df3['Away Rank']<=10.0),1,0)
df3['Top15'] = np.where((df3['Home Rank']<=15.0),1,0)+np.where((df3['Away Rank']<=15.0),1,0)
df3['Top25'] = np.where((df3['Home Rank']<=25.0),1,0)+np.where((df3['Away Rank']<=25.0),1,0)

df3['Real date'] = pd.to_datetime(df3['Dateformat'])
df3['OnThursday'] = np.where((df3['Real date'].dt.dayofweek == 3),1,0)
df3['OnFriday'] = np.where((df3['Real date'].dt.dayofweek == 4),1,0)
df3['OnSaturday'] = np.where((df3['Real date'].dt.dayofweek == 5),1,0)
df3['Hour'] = df3['Time'].astype('str').str.slice(0,2).astype('int')
df3['EarlyAfternoon'] = np.where((df3['Hour']>=6)&(df3['Hour']<15),1,0)
df3['LateAfternoon'] = np.where((df3['Hour']>=15)&(df3['Hour']<18),1,0)
df3['PrimeTime'] = np.where((df3['Hour']>=18)&(df3['Hour']<21),1,0)
df3['LatePrime'] = np.where((df3['Hour']>=21),1,0)

windowDf = pd.DataFrame(np.sort(df3['Dateformat'].unique()),columns=['Date'])
windowDf[['EarlyAfternoon1','EarlyAfternoon2','EarlyAfternoon3','LateAfternoon1','LateAfternoon2','LateAfternoon3','PrimeTime1','PrimeTime2','PrimeTime3','LatePrime1','LatePrime2','LatePrime3']] = [0,0,0,0,0,0,0,0,0,0,0,0]
def collect_windows(r):
  r.EarlyAfternoon1 = df3[(df3['Dateformat'] == r.Date) & (df3['EarlyAfternoon'] == 1) & (df3['AddBin']<=4)]['EarlyAfternoon'].sum()
  r.EarlyAfternoon2 = df3[(df3['Dateformat'] == r.Date) & (df3['EarlyAfternoon'] == 1) & ((df3['AddBin']>=5) & (df3['AddBin']<9))]['EarlyAfternoon'].sum()
  r.EarlyAfternoon3 = df3[(df3['Dateformat'] == r.Date) & (df3['EarlyAfternoon'] == 1) & (df3['AddBin']>8)]['EarlyAfternoon'].sum()
  r.LateAfternoon1 = df3[(df3['Dateformat'] == r.Date) & (df3['LateAfternoon'] == 1) & (df3['AddBin']<=4)]['LateAfternoon'].sum()
  r.LateAfternoon2 = df3[(df3['Dateformat'] == r.Date) & (df3['LateAfternoon'] == 1) & ((df3['AddBin']>=5) & (df3['AddBin']<9))]['LateAfternoon'].sum()
  r.LateAfternoon3 = df3[(df3['Dateformat'] == r.Date) & (df3['LateAfternoon'] == 1) & (df3['AddBin']>8)]['LateAfternoon'].sum()
  r.PrimeTime1 = df3[(df3['Dateformat'] == r.Date) & (df3['PrimeTime'] == 1) & (df3['AddBin']<=4)]['PrimeTime'].sum()
  r.PrimeTime2 = df3[(df3['Dateformat'] == r.Date) & (df3['PrimeTime'] == 1) & ((df3['AddBin']>=5) & (df3['AddBin']<9))]['PrimeTime'].sum()
  r.PrimeTime3 = df3[(df3['Dateformat'] == r.Date) & (df3['PrimeTime'] == 1) & (df3['AddBin']>8)]['PrimeTime'].sum()
  r.LatePrime1 = df3[(df3['Dateformat'] == r.Date) & (df3['LatePrime'] == 1) & (df3['AddBin']<=4)]['LatePrime'].sum()
  r.LatePrime2 = df3[(df3['Dateformat'] == r.Date) & (df3['LatePrime'] == 1) & ((df3['AddBin']>=5) & (df3['AddBin']<9))]['LatePrime'].sum()
  r.LatePrime3 = df3[(df3['Dateformat'] == r.Date) & (df3['LatePrime'] == 1) & (df3['AddBin']>8)]['LatePrime'].sum()
  return r

windowDf = windowDf.apply(collect_windows,axis=1)

def game_competition(r):
  if (r.EarlyAfternoon == 1):
    r.Window_A_Games = windowDf[windowDf['Date'] == r.Dateformat]['EarlyAfternoon3'].sum()
    r.Window_B_Games = windowDf[windowDf['Date'] == r.Dateformat]['EarlyAfternoon2'].sum()
    r.Window_C_Games = windowDf[windowDf['Date'] == r.Dateformat]['EarlyAfternoon1'].sum()
  elif (r.LateAfternoon == 1):
    r.Window_A_Games = windowDf[windowDf['Date'] == r.Dateformat]['LateAfternoon3'].sum()
    r.Window_B_Games = windowDf[windowDf['Date'] == r.Dateformat]['LateAfternoon2'].sum()
    r.Window_C_Games = windowDf[windowDf['Date'] == r.Dateformat]['LateAfternoon1'].sum()
  elif (r.PrimeTime == 1):
    r.Window_A_Games = windowDf[windowDf['Date'] == r.Dateformat]['PrimeTime3'].sum()
    r.Window_B_Games = windowDf[windowDf['Date'] == r.Dateformat]['PrimeTime2'].sum()
    r.Window_C_Games = windowDf[windowDf['Date'] == r.Dateformat]['PrimeTime1'].sum()
  elif (r.LatePrime == 1):
    r.Window_A_Games = windowDf[windowDf['Date'] == r.Dateformat]['LatePrime3'].sum()
    r.Window_B_Games = windowDf[windowDf['Date'] == r.Dateformat]['LatePrime2'].sum()
    r.Window_C_Games = windowDf[windowDf['Date'] == r.Dateformat]['LatePrime1'].sum()
  if (r.AddBin > 8):
    r.Window_A_Games = r.Window_A_Games-1
  elif ((r.AddBin<=8) & (r.AddBin>=5)):
    r.Window_B_Games = r.Window_B_Games-1
  elif (r.AddBin<=4):
    r.Window_C_Games = r.Window_C_Games-1
  return r

df3[['Window_A_Games','Window_B_Games','Window_C_Games']] = [0,0,0]
df3 = df3.apply(game_competition,axis=1)
df3['Games_In_Window'] = df3['Window_A_Games'] + df3['Window_B_Games'] + df3['Window_C_Games']

df4 = df3

XCols = df4.drop(labels=['Line','OULine','HomeGames','AwayGames','Dateformat','Home','Away','Time','Year','Viewers','ViewersK','Network','HomeConf','AwayConf','MultBin','Home Rank','Away Rank','Hour','HomePred','AwayPred','OUPred','Postseason','Real date'],axis=1).keys()
X=df4[XCols]
Y=df4['ViewersK']

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=.3,random_state=364)

model_rfr = RandomForestRegressor()
model_rfr.fit(X_train,Y_train)
Y_pred = model_rfr.predict(X_test)

model_gbr = GradientBoostingRegressor()
model_gbr.fit(X_train,Y_train)
Y_pred_gbr = model_gbr.predict(X_test)

model_etr = ExtraTreesRegressor()
model_etr.fit(X_train,Y_train)
Y_pred_et = model_etr.predict(X_test)

# model_lgbm = LGBMRegressor()
# model_lgbm.fit(X_train,Y_train)
# Y_pred_lgbm = model_lgbm.predict(X_test)

tuned_model_etr = ExtraTreesRegressor(n_estimators=25,criterion='absolute_error')
tuned_model_etr.fit(X_test,Y_test)
Y_pred_et_tune = tuned_model_etr.predict(X_test)

tuned_model_gbr = GradientBoostingRegressor(loss='absolute_error',n_estimators=115,max_depth=2)
tuned_model_gbr.fit(X_test,Y_test)
Y_pred_gbr_tune = tuned_model_gbr.predict(X_test)

tuned_model_etr.fit(X,Y)
tuned_model_gbr.fit(X,Y)

filename2 = 'tuned_gbr.sav'
filename1 = 'tuned_etr.sav'
pickle.dump(tuned_model_etr,open(filename1,'wb'))
pickle.dump(tuned_model_gbr,open(filename2,'wb'))

Y_gbr_tuned = tuned_model_gbr.predict(X)
Y_et_tuned = tuned_model_gbr.predict(X)

df2021 = pd.read_excel('ratings-2021.xlsx')

def collect_windows2(r,dfnew):
  r.EarlyAfternoon1 = dfnew[(dfnew['Dateformat'] == r.Date) & (dfnew['EarlyAfternoon'] == 1) & (dfnew['AddBin']<=4)]['EarlyAfternoon'].sum()
  r.EarlyAfternoon2 = dfnew[(dfnew['Dateformat'] == r.Date) & (dfnew['EarlyAfternoon'] == 1) & ((dfnew['AddBin']>=5) & (dfnew['AddBin']<9))]['EarlyAfternoon'].sum()
  r.EarlyAfternoon3 = dfnew[(dfnew['Dateformat'] == r.Date) & (dfnew['EarlyAfternoon'] == 1) & (dfnew['AddBin']>8)]['EarlyAfternoon'].sum()
  r.LateAfternoon1 = dfnew[(dfnew['Dateformat'] == r.Date) & (dfnew['LateAfternoon'] == 1) & (dfnew['AddBin']<=4)]['LateAfternoon'].sum()
  r.LateAfternoon2 = dfnew[(dfnew['Dateformat'] == r.Date) & (dfnew['LateAfternoon'] == 1) & ((dfnew['AddBin']>=5) & (dfnew['AddBin']<9))]['LateAfternoon'].sum()
  r.LateAfternoon3 = dfnew[(dfnew['Dateformat'] == r.Date) & (dfnew['LateAfternoon'] == 1) & (dfnew['AddBin']>8)]['LateAfternoon'].sum()
  r.PrimeTime1 = dfnew[(dfnew['Dateformat'] == r.Date) & (dfnew['PrimeTime'] == 1) & (dfnew['AddBin']<=4)]['PrimeTime'].sum()
  r.PrimeTime2 = dfnew[(dfnew['Dateformat'] == r.Date) & (dfnew['PrimeTime'] == 1) & ((dfnew['AddBin']>=5) & (dfnew['AddBin']<9))]['PrimeTime'].sum()
  r.PrimeTime3 = dfnew[(dfnew['Dateformat'] == r.Date) & (dfnew['PrimeTime'] == 1) & (dfnew['AddBin']>8)]['PrimeTime'].sum()
  r.LatePrime1 = dfnew[(dfnew['Dateformat'] == r.Date) & (dfnew['LatePrime'] == 1) & (dfnew['AddBin']<=4)]['LatePrime'].sum()
  r.LatePrime2 = dfnew[(dfnew['Dateformat'] == r.Date) & (dfnew['LatePrime'] == 1) & ((dfnew['AddBin']>=5) & (dfnew['AddBin']<9))]['LatePrime'].sum()
  r.LatePrime3 = dfnew[(dfnew['Dateformat'] == r.Date) & (dfnew['LatePrime'] == 1) & (dfnew['AddBin']>8)]['LatePrime'].sum()
  return r

def process_dataset_1(dfnew):
  dfnew = dfnew.dropna(subset=['Viewers','HomeOff','HomeDef','AwayOff','AwayDef'])
  dfnew = dfnew[dfnew['Postseason'] == 0].reset_index(drop=True)
  dfnew[['TV_OTA','TV_ESPN','TV_Cable_ESPN2','TV_Cable_FS1']] = [0,0,0,0]
  dfnew = dfnew.apply(network_grouping, axis=1)
  dfnew['ViewersK'] = dfnew['Viewers'] / 1000
  dfnew['HomeOff'] = dfnew['HomeOff'].replace(0,dfnew['HomeOff'].mean())
  dfnew['HomeDef'] = dfnew['HomeDef'].replace(0,dfnew['HomeDef'].mean())
  dfnew['AwayOff'] = dfnew['AwayOff'].replace(0,dfnew['HomeOff'].mean())
  dfnew['AwayDef'] = dfnew['AwayDef'].replace(0,dfnew['AwayDef'].mean())
  dfnew['HomePred'] = (dfnew['HomeOff']+dfnew['AwayDef'])/2
  dfnew['AwayPred'] = (dfnew['HomeDef']+dfnew['AwayOff'])/2
  dfnew['OUPred'] = dfnew['HomePred'] + dfnew['AwayPred']
  dfnew['HomeBin'] = 0
  dfnew['AwayBin'] = 0
  dfnew = dfnew.apply(add_bins,axis=1)
  dfnew['AddBin'] = dfnew['HomeBin'] + dfnew['AwayBin']
  dfnew['MultBin'] = dfnew['HomeBin'] * dfnew['AwayBin']
  dfnew['HasACC'] = np.where((dfnew['HomeConf']=='ACC') | (dfnew['AwayConf']=='ACC'),1,0)
  dfnew['HasBigTen'] = np.where((dfnew['HomeConf']=='Big Ten') | (dfnew['AwayConf']=='Big Ten'),1,0)
  dfnew['HasBig12'] = np.where((dfnew['HomeConf']=='Big 12') | (dfnew['AwayConf']=='Big 12'),1,0)
  dfnew['HasPac12'] = np.where((dfnew['HomeConf']=='Pac-12') | (dfnew['AwayConf']=='Pac-12'),1,0)
  dfnew['HasSEC'] = np.where((dfnew['HomeConf']=='SEC') | (dfnew['AwayConf']=='SEC'),1,0)
  dfnew['HasFCS'] = np.where((dfnew['HomeConf']=='FCS') | (dfnew['AwayConf']=='FCS'),1,0)
  dfnew['A5vA5'] = np.where(((dfnew['HasACC']+dfnew['HasBigTen']+dfnew['HasBig12']+dfnew['HasPac12']+dfnew['HasSEC']+dfnew['FBSConfGame'])==2),1,0)
  dfnew['A5vG5'] = np.where(((dfnew['HasACC']+dfnew['HasBigTen']+dfnew['HasBig12']+dfnew['HasPac12']+dfnew['HasSEC'])==1)&((dfnew['FBSConfGame']+dfnew['HasFCS'])==0),1,0)
  dfnew['G5vG5'] = np.where(((dfnew['HasACC']+dfnew['HasBigTen']+dfnew['HasBig12']+dfnew['HasPac12']+dfnew['HasSEC'])==0)&(dfnew['HasFCS']==0),1,0)
  dfnew['Top2'] = np.where((dfnew['Home Rank']<=2.0),1,0)+np.where((dfnew['Away Rank']<=2.0),1,0)
  dfnew['Top5'] = np.where((dfnew['Home Rank']<=5.0),1,0)+np.where((dfnew['Away Rank']<=5.0),1,0)
  dfnew['Top10'] = np.where((dfnew['Home Rank']<=10.0),1,0)+np.where((dfnew['Away Rank']<=10.0),1,0)
  dfnew['Top15'] = np.where((dfnew['Home Rank']<=15.0),1,0)+np.where((dfnew['Away Rank']<=15.0),1,0)
  dfnew['Top25'] = np.where((dfnew['Home Rank']<=25.0),1,0)+np.where((dfnew['Away Rank']<=25.0),1,0)
  dfnew['Real date'] = pd.to_datetime(dfnew['Dateformat'])
  dfnew['OnThursday'] = np.where((dfnew['Real date'].dt.dayofweek == 3),1,0)
  dfnew['OnFriday'] = np.where((dfnew['Real date'].dt.dayofweek == 4),1,0)
  dfnew['OnSaturday'] = np.where((dfnew['Real date'].dt.dayofweek == 5),1,0)
  dfnew['Hour'] = dfnew['Time'].astype('str').str.slice(0,2).astype('int')
  dfnew['EarlyAfternoon'] = np.where((dfnew['Hour']>=6)&(dfnew['Hour']<15),1,0)
  dfnew['LateAfternoon'] = np.where((dfnew['Hour']>=15)&(dfnew['Hour']<18),1,0)
  dfnew['PrimeTime'] = np.where((dfnew['Hour']>=18)&(dfnew['Hour']<21),1,0)
  dfnew['LatePrime'] = np.where((dfnew['Hour']>=21),1,0)
  return dfnew

df2021 = df2021[df2021['Network'].isin(['ABC','CBS','NBC','FOX','ESPN','ESPN2','FS1'])]

df2021['HomeGames'] = df2021['HomeWins']+df2021['HomeLosses']
df2021['AwayGames'] = df2021['AwayWins']+df2021['AwayLosses']
df2021['RecDifferentials'] = (df2021['HomeWins'] - df2021['HomeLosses']) + (df2021['AwayWins'] - df2021['AwayLosses'])

df2021 = process_dataset_1(df2021)

windowDf = pd.DataFrame(np.sort(df2021['Dateformat'].unique()),columns=['Date'])
windowDf[['EarlyAfternoon1','EarlyAfternoon2','EarlyAfternoon3','LateAfternoon1','LateAfternoon2','LateAfternoon3','PrimeTime1','PrimeTime2','PrimeTime3','LatePrime1','LatePrime2','LatePrime3']] = [0,0,0,0,0,0,0,0,0,0,0,0]
windowDf = windowDf.apply(collect_windows2,args=(df2021,),axis=1)
df2021[['Window_A_Games','Window_B_Games','Window_C_Games']] = [0,0,0]
df2021 = df2021.apply(game_competition,axis=1)
df2021['Games_In_Window'] = df2021['Window_A_Games'] + df2021['Window_B_Games'] + df2021['Window_C_Games']

X2=df2021[XCols]
Y2=df2021['ViewersK']

Y_pred2 = model_etr.predict(X2)

Y_pred3 = tuned_model_etr.predict(X2)

# Y_pred4 = model_lgbm.predict(X2)

Y_pred5 = model_gbr.predict(X2)
Y_pred6 = tuned_model_gbr.predict(X2)

df2021['pred'] = (Y_pred6 + Y_pred3) / 2
df2021['pred_gbr'] = Y_pred6
df2021['pred_etr'] = Y_pred3
df2021['pred_model_diff'] = ((df2021['pred_gbr'] - df2021['pred_etr']) / (df2021['pred_gbr']+df2021['pred_etr']))*100

df2021['diff'] = df2021['ViewersK'] - df2021['pred']

df2021['diffpct'] = (abs(df2021['diff'] / df2021['pred'])*100).astype('int')

df2021.to_csv('2021withpredictions.csv')

df2022 = pd.read_excel('ratings-2022.xlsx')

df2022 = df2022[df2022['Network'].isin(['ABC','CBS','NBC','FOX','ESPN','ESPN2','FS1'])]
df2022['HomeGames'] = df2022['HomeWins']+df2022['HomeLosses']
df2022['AwayGames'] = df2022['AwayWins']+df2022['AwayLosses']
df2022['RecDifferentials'] = (df2022['HomeWins'] - df2022['HomeLosses']) + (df2022['AwayWins'] - df2022['AwayLosses'])
df2022 = process_dataset_1(df2022)

windowDf = pd.DataFrame(np.sort(df2022['Dateformat'].unique()),columns=['Date'])
windowDf[['EarlyAfternoon1','EarlyAfternoon2','EarlyAfternoon3','LateAfternoon1','LateAfternoon2','LateAfternoon3','PrimeTime1','PrimeTime2','PrimeTime3','LatePrime1','LatePrime2','LatePrime3']] = [0,0,0,0,0,0,0,0,0,0,0,0]
windowDf = windowDf.apply(collect_windows2,args=(df2022,),axis=1)
df2022[['Window_A_Games','Window_B_Games','Window_C_Games']] = [0,0,0]
df2022 = df2022.apply(game_competition,axis=1)
df2022['Games_In_Window'] = df2022['Window_A_Games'] + df2022['Window_B_Games'] + df2022['Window_C_Games']

X4=df2022[XCols]
Y4=df2022['ViewersK']

y2022_pred_gbr = tuned_model_gbr.predict(X4)
y2022_pred_etr = tuned_model_etr.predict(X4)
df2022['pred'] = (y2022_pred_gbr + y2022_pred_etr) / 2
df2022['pred'] = round(df2022['pred'],1)
df2022['diff'] = round(df2022['ViewersK'] - df2022['pred'],0)
df2022['diffpct'] = (abs(df2022['diff'] / df2022['pred'])*100).astype('int')

df2022.to_csv('2022withpredictions.csv')

print(df2022[df2022['Viewers']>0][['Dateformat','Time','Network','Home','Away','ViewersK','AddBin','pred','diff','diffpct']])

print(df2022[df2022['Viewers']>0]['diffpct'].mean())

print(df2022[df2022['Viewers']==0][['Dateformat','Time','Network','Home','Away','ViewersK','AddBin','pred']])