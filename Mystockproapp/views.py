# from re import S
# from tkinter import N
from django.shortcuts import render, HttpResponse, HttpResponseRedirect
import MySQLdb
import tweepy
from textblob import TextBlob
import matplotlib.pyplot as plot
import numpy as np
import requests
from keras.models import Sequential
from keras.layers import Dense
from sklearn.svm import SVR
import csv
from django.core.files.storage import *

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
db = MySQLdb.connect("localhost", "root", "", "mystock")
c = db.cursor()


def index(request):
    msg = ""
    if "b1" in request.POST:
        uname = request.POST.get("t1")
        email = request.POST.get("t2")
        password = request.POST.get("t3")
        c.execute("select count(*) from registration  where uname='"+str(uname)+"'")
        data1 = c.fetchone()
        if (int(data1[0]) == 0):

            c.execute("insert into registration values('" +
                      str(uname)+"','"+str(email)+"','"+str(password)+"')")
            db.commit()
        else:
            msg = "username already exist"
    if "b2" in request.POST:
        uname = request.POST.get("t1")
        password = request.POST.get("t2")
        c.execute("select count(*) as cnt from registration where uname='" +
                  str(uname)+"' and password='"+str(password)+"'")
        data = c.fetchone()
        request.session["uname"] = uname
        if (data):
            if (int(data[0]) == 0):
                msg = "invalid username or password"
            else:
                return HttpResponseRedirect('/userHome')
        else:
            msg = "invalid username or password"

    return render(request, "index.html", {"msg": msg})


def userHome(request):
    c.execute("select * from company")
    img=c.fetchall()
    print(img[0][0])
    return render(request, "userhome.html",{"img":img})


def myprofile(request):
    msg = ""
    uname = request.session.get("uname")
    qry = "select * from registration where uname='"+str(uname)+"'"
    c.execute(qry)
    data = c.fetchone()

    if request.POST:
        uname = request.POST.get("t1")
        email = request.POST.get("t2")
        password = request.POST.get("t3")
        qry = "update registration set email='" + \
            str(email)+"',password='"+str(password) + \
            "' where uname='"+str(uname)+"'"
        print(qry)
        c.execute(qry)
        db.commit()
        msg = "updated successfully"
    return render(request, "myprofile.html", {"data": data, "msg": msg})


def Sharedetails(request):
    c.execute("select * from Company")
    data=c.fetchall()
    return render(request, "Sharedetails.html",{"data":data})


def predictprices(request):
    c.execute("select * from Company")
    datas=c.fetchall()
    prices = ""
    start = ""
    ends = ""
    data = ""
    if request.POST:

        data = request.POST.get("stock")

        date = request.POST.get("t1")
        # if gethistoricaldata(data):
        # msg="No data found"
        gethistoricaldata(data)

        res = stockprediction()
        print("##########################################################################################")
        print(res)
        prices = res["value"]
        print(prices)
        start = res["result"][0]
        ends = res["result"][1]
        print(start)
        print(ends)

        # dates=["10","11","12"]
        # result=predictprice(dates,prices,29)
        # print(result)
        print(
            "****************************************************************************")

    return render(request, "predictprice.html", {"price": prices, "start": start, "ends": ends, "data": data,"datas":datas})


ConsumerKey = "zS2ibMBtTpsDu8ekzIKJva8Mj"
ConsumerSecret = "EvsXL1zBLwFo78lHqRXI8Fv3N33uQq0w8620zyCxrE8WesFjVT"
AccessToken = "3041974901-ZRJqJQOUFzt4yiS3KGtd3CBnAocxh1NlUSNIC0x"
AccessTokenSecret = "zOOYb6u9OrWqz2mczVwOPnKAPvDfFappb2lR0VrrbzB1W"
auth = tweepy.OAuthHandler(ConsumerKey, ConsumerSecret)
auth.set_access_token(AccessToken, AccessTokenSecret)
# con=tweepy.API(auth)
# tweets=con.search("oru_adaar_love")
# app=Flask("__name__")


filename = "sto.csv"

dates = []
prices = []


def gethistoricaldata(histo):
    # url='https://finance.google.com/finance/historical?q= %3A'+histo+'&output=csv'
    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY_ADJUSTED&symbol=' + \
        histo+'&apikey=1ZW64R971VO4W8SH&datatype=csv'
    print(url)
    r = requests.get(url, stream=True)
    if r.status_code != 400:
        with open(filename, "wb") as f:
            for line in r:
                f.write(line)

        return True

        # print(getNews(x))


def stockprediction():
    dataset = []
    with open(filename) as f:
        for n, line in enumerate(f):
            if n != 0:
                dataset.append(float(line.split(',')[1]))
        dataset = np.array(dataset)

        def createdataset(dataset):
            datax = [dataset[n+1] for n in range(len(dataset)-2)]
            return np.array(datax), dataset[2:]
        trainx, trainy = createdataset(dataset)
        model = Sequential()  # Process 1st layer then 2nd
        # ann model 8 layers, input_diamention 1(one diamentional array)
        model.add(Dense(8, input_dim=1, activation='relu'))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(trainx, trainy, epochs=20, batch_size=2, verbose=2)
        prediction = model.predict(np.array([dataset[0]]))
        result = [dataset[0], prediction[0, 0]]
        print(result)
        dictionary = {"value": str(prediction[0, 0]), "result": result}
        return (dictionary)


def getdata(filename):
    with open(filename, 'r') as csvfile:
        csvfilereader = csv.reader(csvfile)
        next(csvfilereader)
        for raw in csvfilereader:
            dates.append(int(raw[0].split('-')[0]))
            prices.append(float(raw[1]))
    return


def predictprice(dates, prices, x):
    dates = np.reshape(dates, (len(dates), 1))
    svrline = SVR(kernel='linear', C=1e3)
    svrpoly = SVR(kernel='poly', C=1e3, degree=2)
    svr_rpf = SVR(kernel='rbf', C=1e3, gamma=.1)
    svrline.fit(dates, prices)
    svrpoly.fit(dates, prices)
    svr_rpf.fit(dates, prices)
    plot.scatter(dates, prices, color='black', label='DATA')
    plot.plot(dates, svr_rpf.predict(dates), color='red', label='rbf_model')
    plot.plot(dates, svrline.predict(dates), color='blue', label='line_model')
    plot.plot(dates, svrpoly.predict(dates), color='green', label='ploy_model')
    plot.xlabel('DATE')
    plot.ylabel('price')
    plot.title("Stock Prediction")
    plot.legend()
    plot.show()
    return svr_rpf.predict(x)[0], svrline.predict(x)[0], svrpoly.predict(x)[0]

def AddCompany(request):
    msg=""
    if request.POST:
        name=request.POST["t1"]
        symbol=request.POST["t2"]
        img = request.FILES.get("img")
        fs = FileSystemStorage()
        filename = fs.save(img.name, img)
        path = fs.url(filename)
        c.execute("insert into company (name,symbol,image) values('"+str(name)+"','"+str(symbol)+"','"+str(path)+"')")
        db.commit()
    return render(request,"AddCompany.html")

def stockanalysis(request):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    # Importing all necessary libraries.
    data = request.GET.get("data")
    # Using data from Apple's stock.
    df = pd.read_csv('sto.csv')
    df.head()
    df.info()
    df.describe()
    df.shape
    # Visualizing the opening prices of the data.
    plt.figure(figsize=(16, 8))
    plt.title(data)
    plt.xlabel('Days')
    plt.ylabel('Opening Price USD ($)')
    plt.plot(df['open'])
    plt.show()

    # Visualizing the high prices of the data.
    plt.figure(figsize=(16, 8))
    plt.title(data)
    plt.xlabel('Days')
    plt.ylabel('High Price USD ($)')
    plt.plot(df['high'])
    plt.show()

    # Visualizing the low prices of the data.
    plt.figure(figsize=(16, 8))
    plt.title(data)
    plt.xlabel('Days')
    plt.ylabel('Low Price USD ($)')
    plt.plot(df['low'])
    plt.show()

    # Visualizing the closing prices of the data.
    plt.figure(figsize=(16, 8))
    plt.title(data)
    plt.xlabel('Days')
    plt.ylabel('Closing Price USD ($)')
    plt.plot(df['close'])
    plt.show()

    df2 = df['close']
    df2.tail()

    df2 = pd.DataFrame(df2)
    df2.tail()

    # Prediction 100 days into the future.
    future_days = 10
    df2['Prediction'] = df2['close'].shift(-future_days)
    print(df2.tail())

    X = np.array(df2.drop(['Prediction'], 1))[:-future_days]
    print(X)

    y = np.array(df2['Prediction'])[:-future_days]
    print(y)
    print("##################################################################################")

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Implementing Linear and Decision Tree Regression Algorithms.
    tree = DecisionTreeRegressor().fit(x_train, y_train)
    lr = LinearRegression().fit(x_train, y_train)
    x_future = df2.drop(['Prediction'], 1)[:-future_days]
    x_future = x_future.tail(future_days)
    x_future = np.array(x_future)
    x_future
    tree_prediction = tree.predict(x_future)
    print(tree_prediction)

    lr_prediction = lr.predict(x_future)
    print(lr_prediction)

    predictions = tree_prediction
    valid = df2[X.shape[0]:]
    valid['Predictions'] = predictions

    plt.figure(figsize=(16, 8))
    plt.title("Model")
    plt.xlabel('Days')
    plt.ylabel('Close Price USD ($)')
    plt.plot(df2['close'])
    plt.plot(valid[['close', 'Predictions']])
    plt.legend(["Original", "Valid", 'Predicted'])
    plt.show()

    return render(request, "stockanalysis.html")
