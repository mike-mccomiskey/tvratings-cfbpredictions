import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor
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

filename2 = 'tuned_gbr.sav'
filename1 = 'tuned_etr.sav'
tuned_model_etr = (pickle.load(open(filename1,'rb')))
tuned_model_gbr = (pickle.load(open(filename2,'rb')))

lookupDf = pd.read_csv('lookupdf.csv')
# Y_gbr_tuned = tuned_model_gbr.predict(X)
# Y_et_tuned = tuned_model_gbr.predict(X)

def add_bins(r):
  r.HomeBin = lookupDf[lookupDf['Teams']==r.Home]['bins'].sum()
  r.AwayBin = lookupDf[lookupDf['Teams']==r.Away]['bins'].sum()
  return r

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

XCols = df2022.drop(labels=['Line','OULine','HomeGames','AwayGames','Dateformat','Home','Away','Time','Year','Viewers','ViewersK','Network','HomeConf','AwayConf','MultBin','Home Rank','Away Rank','Hour','HomePred','AwayPred','OUPred','Postseason','Real date'],axis=1).keys()

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