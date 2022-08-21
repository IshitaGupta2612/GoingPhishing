
from flask import Flask,render_template,request,redirect,Response, send_file, make_response, url_for
from flask.wrappers import Request
import pandas as pd
from twilio.rest import Client
import numpy as np
import datetime as dt
import io
import random
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


plt.style.use('ggplot')
import matplotlib.style as style

import seaborn as sns
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

from sklearn import preprocessing
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

data = pd.read_csv('final_ip_mapped_data.csv')
data['@timestamp'] = pd.to_datetime(data['@timestamp'])
data.sort_values(['ip_address', '@timestamp'], inplace=True)
data['shift_time'] = data.groupby(['ip_address'])['@timestamp'].shift(1)
data['time_diff'] = (data['@timestamp'] - data['shift_time']).dt.seconds//60
data['date'] = data['@timestamp'].dt.date
data['dow'] = data['@timestamp'].dt.weekday
data['hour'] = data['@timestamp'].dt.hour
data['is_weekend'] = ((data['dow']==5)|(data['dow']==6)).astype(int)
data['hour_bucket'] = data['hour']//4
ip_col = 'ip_address'
ip_counts = data.groupby(ip_col)['@timestamp'].count().reset_index()
ip_counts = ip_counts.rename(columns={'@timestamp':'total_count'})
daily_counts = data.groupby([ip_col, 'date'])['@timestamp'].count().reset_index()
daily_counts = daily_counts.rename(columns={'@timestamp':'daily_counts'})
daily_counts_agg = daily_counts.groupby(ip_col).daily_counts.median().reset_index()

weekend_counts = data.groupby([ip_col, 'is_weekend'])['@timestamp'].count().reset_index()
weekend_counts = weekend_counts.rename(columns={'@timestamp':'weekend_counts'})
weekend_counts_agg = weekend_counts.pivot_table(index=ip_col, columns='is_weekend').reset_index([0])
weekend_counts_agg.columns = weekend_counts_agg.columns.droplevel()
weekend_counts_agg.columns = [ip_col, 'week_day', 'weekend']
weekend_counts_agg['is_weekend_ratio'] = weekend_counts_agg['week_day']/ weekend_counts_agg['weekend']
lean_weekend_counts_agg = weekend_counts_agg[[ip_col, 'is_weekend_ratio']]

avg_timedelta_data = data.groupby(ip_col).agg({'time_diff':['mean','max']}).reset_index()
avg_timedelta_data.columns = avg_timedelta_data.columns.droplevel()
avg_timedelta_data.columns = [ip_col, 'td_mean', 'td_max']

merge_1 = ip_counts.merge(daily_counts_agg, on=ip_col, how='left')
merge_2 = merge_1.merge(lean_weekend_counts_agg, on=ip_col, how='left')
final_data = merge_2.merge(avg_timedelta_data, on=ip_col, how='left')
df=final_data.head(10)

app = Flask(__name__)


@app.route('/')
def hello_world():
    maxOccuringIP=ip_counts['ip_address'].loc[ip_counts['total_count']==ip_counts['total_count'].max()]
    maxOccuringIP=np.array(maxOccuringIP)
    maxOccuringIP=maxOccuringIP[0]

    highestDailyCount=daily_counts_agg['ip_address'].loc[daily_counts_agg['daily_counts']==daily_counts_agg['daily_counts'].max()]
    highestDailyCount=np.array(highestDailyCount)
    highestDailyCount=highestDailyCount[0]

    ipHighestWeekendRatio=lean_weekend_counts_agg['ip_address'].loc[lean_weekend_counts_agg['is_weekend_ratio']==lean_weekend_counts_agg['is_weekend_ratio'].max()]
    ipHighestWeekendRatio=np.array(ipHighestWeekendRatio)
    ipHighestWeekendRatio=ipHighestWeekendRatio[0]

    higestAvgLoginTime=avg_timedelta_data['ip_address'].loc[avg_timedelta_data['td_mean']==avg_timedelta_data['td_mean'].max()]
    higestAvgLoginTime=np.array(higestAvgLoginTime)
    higestAvgLoginTime=higestAvgLoginTime[0]

    



    return render_template('index.html', maxOccuringIP=maxOccuringIP,highestDailyCount=highestDailyCount,ipHighestWeekendRatio=ipHighestWeekendRatio,higestAvgLoginTime=higestAvgLoginTime, tables=[df.to_html(classes='data')], titles=df.columns.values)



@app.route('/mostIP.png')
def plot_png():
    x=ip_counts.sort_values('total_count', ascending=False)
    x=x.iloc[:10]
    lst1=x['ip_address'].to_numpy()
    lst2=x['total_count'].to_numpy()

    fig, ax = plt.subplots(figsize =(9, 9))
    ax.barh(lst1, lst2)
    ax.invert_yaxis()
    for i in ax.patches:
        plt.text(i.get_width()+0.2, i.get_y()+0.5,str(round((i.get_width()), 2)),fontsize = 9,color ='black')

    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
 
    plt.xlabel('Occurences', fontdict={'fontsize': 9})
    plt.ylabel('IP Addresses', fontdict={'fontsize': 9})
    plt.xlim([4200, 4375])
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/dailyIP.png')
def plot_png2():
    x1=daily_counts_agg.sort_values('daily_counts', ascending=False)
    x1=x1.iloc[:10]
    lst3=x1['ip_address'].to_numpy()
    lst4=x1['daily_counts'].to_numpy()
    fig, ax = plt.subplots(figsize =(9, 9))
 
    # Horizontal Bar Plot
    ax.barh(lst3, lst4)
    # Show top values
    ax.invert_yaxis()
 
    # Add annotation to bars
    for i in ax.patches:
        plt.text(i.get_width()+0.2, i.get_y()+0.5,
                 str(round((i.get_width()), 2)),
                 fontsize = 15,
                color ='black')
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9)


    # Show Plot
    plt.xlabel('Daily Count', fontdict={'fontsize': 9})
    plt.ylabel('IP Addresses', fontdict={'fontsize': 9})
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/weekendIP.png')
def plot_png3():
    x3=lean_weekend_counts_agg.sort_values('is_weekend_ratio', ascending=False)
    x3=x3.iloc[:10]
    lst5=x3['ip_address'].to_numpy()
    lst6=x3['is_weekend_ratio'].to_numpy()
    #lst2=lst2.astype(int)

    fig, ax = plt.subplots(figsize =(9, 9))
 
    # Horizontal Bar Plot
    ax.barh(lst5, lst6)
    # Show top values
    ax.invert_yaxis()
 
    # Add annotation to bars
    for i in ax.patches:
        plt.text(i.get_width()+0.2, i.get_y()+0.5,
                str(round((i.get_width()), 2)),
                fontsize = 15,
                color ='black')
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9)

 
    # Show Plot
    plt.xlabel('Weekend Ratio', fontdict={'fontsize': 9})
    plt.ylabel('IP Addresses', fontdict={'fontsize': 9})

    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/timeIP.png')
def plot_png4():
    x4=avg_timedelta_data.sort_values('td_mean', ascending=False)
    x4=x4.iloc[:10]
    lst7=x4['ip_address'].to_numpy()
    lst8=x4['td_mean'].to_numpy()
    #lst2=lst2.astype(int)


    fig, ax = plt.subplots(figsize =(9, 9))
 
    # Horizontal Bar Plot
    ax.barh(lst7, lst8)
    # Show top values
    ax.invert_yaxis()
 
    # Add annotation to bars
    for i in ax.patches:
        plt.text(i.get_width()+0.2, i.get_y()+0.5,
                 str(round((i.get_width()), 2)),
                 fontsize = 15,
                color ='black')

    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
  


 
    # Show Plot
    plt.xlabel('Average Login Time', fontdict={'fontsize': 9})
    plt.ylabel('IP Addresses', fontdict={'fontsize': 9})
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


@app.route('/TwilioForm',methods=['GET','POST'])
def TwilioForm():
    if request.method=="POST":
        account_sid=request.form['sid']
        auth_token=request.form['token']
        twwpfrom=request.form['wpfrom']
        twwpto=request.form['wpto']
        twmsg=request.form['msg']


        client = Client(account_sid, auth_token)

        message = client.messages \
        .create(
            body=twmsg,
            from_='whatsapp:'+twwpfrom,
            to='whatsapp:'+twwpto
        )

        print(message.sid)
        

    return render_template('TwilioForm.html')

# main driver function
if __name__ == '__main__':
	app.run()
