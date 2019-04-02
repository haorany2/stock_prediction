import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
%matplotlib inline
from mpl_finance import candlestick_ohlc
from mpl_finance import candlestick2_ohlc
from mpl_finance import volume_overlay2
from mpl_finance import volume_overlay
from mpl_finance import index_bar
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM,Input
df=pd.read_csv('tw_spydata_raw.csv')
#print the head
#df.head()



#close price and volume will be used in lstm
df['Trade Close']
df['Trade Volume']
#check is there any nan item 
df.isnull().values.sum()

#lstm data
#lstm_data=pd.DataFrame(index=range(0,len(df)),columns=['Close','Volume'])

#for i in range(0,len(df)):
#    lstm_data['Volume'][i]=df['Trade Volume'][i]
#    lstm_data['Close'][i]=df['Trade Close'][i]

lstm_data=df[['Trade Close', 'Trade Volume']].copy()
print("lstm_data: ",lstm_data.shape)
feature_lst=[]
target_lst=[]
lstm_data=lstm_data.values.astype(float)

lstm_data_reducer=lstm_data[1:]
lstm_data_reduce=lstm_data[:-1]#latter, use lstm_data_reducer/lstm_data_reduce
log_return_fea=np.log(lstm_data_reducer)-np.log(lstm_data_reduce)

lstm_data_reducer=lstm_data[34:,0]
lstm_data_reduce=lstm_data[:-34,0]
log_return_tar=np.log(lstm_data_reducer)-np.log(lstm_data_reduce)
print("log_return_tar:",log_return_tar.shape)
#log_return_tar=log_return_fea[29:]
for i in range(29,log_return_fea.shape[0]-4):
    feature_lst.append(log_return_fea[i-29:i])
    target_lst.append(log_return_tar[i-29])
    vol=df.loc[i-29:i,'Trade Volume'].copy()
    opens=df.loc[i-29:i,'Trade Open'].copy()
    closes=df.loc[i-29:i,'Trade Close'].copy()
    highs=df.loc[i-29:i,'Trade High'].copy()
    lows=df.loc[i-29:i,'Trade Low'].copy()
    
    fig = plt.figure()
    #ax = plt.subplot2grid((6,1), (0,0), rowspan = 3, colspan = 1)
    #ax2 = plt.subplot2grid((6,1), (3,0), rowspan = 3, colspan = 1, sharex = ax)
    ax=plt.subplot(211,frameon=False)


    ax2=plt.subplot(212, sharex=ax,frameon=False)
    candlestick2_ohlc(ax, opens.values,highs.values,lows.values,closes.values, width=0.8, colorup='blue', colordown='red')
    
    bc = volume_overlay(ax2, opens.values,closes.values,vol.values, colorup='g', alpha=1, width=0.8,colordown='g')
    
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig('./candle_img/candle'+str(i-29)+'.png')
    plt.cla()
    plt.close()
