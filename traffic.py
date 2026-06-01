import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
df= pd.read_csv("traffic.csv")

df.drop("ID",axis=1,inplace=True)

df["DateTime"]= pd.to_datetime(df["DateTime"])

df["hour"] = df["DateTime"].dt.hour
df["day"] = df["DateTime"].dt.day
df["weekday"] = df["DateTime"].dt.weekday

df.drop("DateTime",inplace=True,axis=1)

df["traffic_level"]=pd.cut(
    df["Vehicles"],
    bins=[0,20,40,300],
    labels=[0,1,2,]
)

x=df.drop("traffic_level",axis=1)
y=df["traffic_level"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
model=dt.fit(x_train,y_train)
model.score(x_test,y_test)


y_pred=model.predict(x_test)
print(classification_report(y_test,y_pred))


print(confusion_matrix(y_test,y_pred))