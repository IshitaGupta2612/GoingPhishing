
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
df=final_data.head(7)

ip_map = final_data[ip_col].to_dict()
RANDOM_STATE = 123
feature_cols = ['total_count', 'daily_counts', 'is_weekend_ratio', 'td_mean', 'td_max']
data_new = final_data[feature_cols]
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(data_new)
data_new = pd.DataFrame(np_scaled, columns=feature_cols)
n_cluster = range(1, 15)
kmeans = [KMeans(n_clusters=i, random_state=RANDOM_STATE).fit(data_new) for i in n_cluster]
scores = [kmeans[i].score(data_new) for i in range(len(kmeans))]
cluster_model = kmeans[5]
final_data['cluster'] = cluster_model.predict(data_new)
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300, random_state=RANDOM_STATE)
tsne_results = tsne.fit_transform(data_new)
final_data['tsne-2d-one'] = tsne_results[:,0]
final_data['tsne-2d-two'] = tsne_results[:,1]
tsne_cluster = final_data.groupby('cluster').agg({'tsne-2d-one':'mean', 'tsne-2d-two':'mean'}).reset_index()
centers = cluster_model.cluster_centers_
points = np.asarray(data_new)
total_distance = pd.Series()
outlier_fraction = 0.028
model =  IsolationForest(n_jobs=-1, n_estimators=200, max_features=3, random_state=RANDOM_STATE, contamination=outlier_fraction)
model.fit(data_new)
# add the data to the main  
final_data['anomaly_isolated'] = pd.Series(model.predict(data_new))
final_data['anomaly_isolated'] = final_data['anomaly_isolated'].map( {1: 0, -1: 1} )
anomalyCount=final_data['anomaly_isolated'].value_counts()
total_counts_condition = (final_data['total_count'] >= final_data['total_count'].quantile(0.98)) 
daily_counts_condition =  (final_data['daily_counts'] >= final_data['daily_counts'].quantile(0.98))
final_data['anomaly_manual'] = (total_counts_condition | daily_counts_condition).astype(np.int)
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, confusion_matrix
def get_sensitivity_specificity(y_true, y_pred):
    cf = confusion_matrix(y_true, y_pred)
    sensitivity = cf[0,0]/(cf[:,0].sum())
    specificity = cf[1,1]/(cf[:,1].sum())
    return sensitivity, specificity
f1_iso = f1_score(final_data['anomaly_manual'],final_data['anomaly_isolated'])
acc_iso = accuracy_score(final_data['anomaly_manual'],final_data['anomaly_isolated'])
roc_iso = roc_auc_score(final_data['anomaly_manual'],final_data['anomaly_isolated'])
sen_iso, spec_iso = get_sensitivity_specificity(final_data['anomaly_manual'],final_data['anomaly_isolated'])
met_iso = {
           'f1_score': f1_iso,
           'accuracy': acc_iso,
           'roc_score': roc_iso,
           'sensitivity': sen_iso,
           'specificity': spec_iso
          }
metrics = {'isolated_forest': met_iso,
          }
metrics_df = pd.DataFrame.from_dict(metrics)




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

@app.route('/tables')
def tableDisplay():
    temp=final_data
    del temp["total_count"]
    met_iso2 = {
           'f1_score': [f1_iso],
           'accuracy': [acc_iso],
           'roc_score': [roc_iso],
           'sensitivity': [sen_iso],
           'specificity': [spec_iso]
          }
    met_iso2=pd.DataFrame(met_iso2)
    return render_template('general-table.html',
                           tables=[ip_counts.head(6).to_html(classes='data', index = False),
                           daily_counts.head(6).to_html(classes='data', index = False),
                           lean_weekend_counts_agg.head(6).to_html(classes='data', index = False),
                           avg_timedelta_data.head(6).to_html(classes='data', index = False),
                           final_data.head(6).to_html(classes='data', index = False),
                           met_iso2.to_html(classes='data', index = False)],
                           titles= ['na','IP Address with their total counts',
                           'IP Addresses with their Daily Counts',
                           'IP Addresses with their Weekend Ratios',
                           'IP Addresses with their Average Time',
                           'Final Input Insights',
                           'Algorithm Accuracy'])

@app.route('/finalGraph1.png')
def plot_pngFinal1():
    fig, ax = plt.subplots(figsize =(4, 4))
    v = np.array(anomalyCount)
    labels =  ['0','1']
    colors = ["#1abc9c", "#2c3e50"]
    explode = [0,0.1]
    wedge_properties = {"edgecolor":"k",'linewidth': 2}

    plt.pie(v, labels=labels, explode=explode, colors=colors, startangle=30,
           counterclock=False, shadow=True, wedgeprops=wedge_properties,
           autopct="%1.1f%%", pctdistance=0.7, textprops={'fontsize': 14})

    plt.title("Anomaly Detection",fontsize=15)
    plt.legend(title='Anomaly Isolated',fontsize=15)    
 
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/finalGraph2.png')
def plot_pngFinal2():
    fig, ax = plt.subplots(figsize =(7,5))
    sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="cluster",
    palette=sns.color_palette("hls", 6),
    data=final_data,
    legend="full",
    alpha=1
    )

    plt.scatter(x="tsne-2d-one", y="tsne-2d-two", data=tsne_cluster, s=100, c='b')

    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/finalGraph3.png')
def plot_pngFinal3():
    fig, ax = plt.subplots(figsize =(7,5))
    sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="anomaly_isolated",
    data=final_data,
    legend="full",
    alpha=1
    )
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')




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
