
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
x=ip_counts.sort_values('total_count', ascending=False)
x=x.iloc[:10]
lst1=x['ip_address'].to_numpy()
lst2=x['total_count'].to_numpy()


app = Flask(__name__)


@app.route('/')
def hello_world():
	return render_template('index.html')

@app.route('/mostIP.png')
def plot_png():
    fig, ax = plt.subplots(figsize =(9, 9))
    ax.barh(lst1, lst2)
    ax.invert_yaxis()
    for i in ax.patches:
        plt.text(i.get_width()+0.2, i.get_y()+0.5,str(round((i.get_width()), 2)),fontsize = 9,color ='black')

    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9)

    plt.title('Top 10 Most Occuring IP Addresses', fontdict={'fontsize': 9})
 
    plt.xlabel('Occurences', fontdict={'fontsize': 9})
    plt.ylabel('IP Addresses', fontdict={'fontsize': 9})
    plt.xlim([4200, 4375])
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

daily_counts = data.groupby([ip_col, 'date'])['@timestamp'].count().reset_index()
daily_counts = daily_counts.rename(columns={'@timestamp':'daily_counts'})
daily_counts_agg = daily_counts.groupby(ip_col).daily_counts.median().reset_index()
x=daily_counts_agg.sort_values('daily_counts', ascending=False)
x=x.iloc[:10]


lst1=x['ip_address'].to_numpy()
lst2=x['daily_counts'].to_numpy()

@app.route('/dailyIP.png')
def plot_png2():
    fig, ax = plt.subplots(figsize =(6, 4))
    # Horizontal Bar Plot
    ax.barh(lst1, lst2)
    # Show top values
    ax.invert_yaxis()
 
    # Add annotation to bars
    for i in ax.patches:
        plt.text(i.get_width()+0.2, i.get_y()+0.5,str(round((i.get_width()), 2)),fontsize = 15,color ='black')

    ax.tick_params(axis="x", labelsize=15)
    ax.tick_params(axis="y", labelsize=15)
    # Add Plot Title
    plt.title('Top 10 IP Addresses with Highest Daily Counts', fontdict={'fontsize': 30})
 
    # Show Plot
    plt.xlabel('Daily Count', fontdict={'fontsize': 20}, labelpad=15)
    plt.ylabel('IP Addresses', fontdict={'fontsize': 20}, labelpad=15)

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
