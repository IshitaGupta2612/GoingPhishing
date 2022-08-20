
from flask import Flask,render_template,request,redirect
from flask.wrappers import Request
from flask_sqlalchemy import SQLAlchemy
import sqlite3 as sql
import pandas as pd
from twilio.rest import Client

app = Flask(__name__)


@app.route('/')
def hello_world():
	return render_template('index.html')

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
