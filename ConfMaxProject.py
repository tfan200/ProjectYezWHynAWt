from numpy.ma.core import correlate
from google.colab import files 
dataToUpload = files.upload()

import pandas as pd
import statistics
import plotly.graph_objects as go 
import plotly.figure_factory as ff
import plotly.express as px
import random 
import numpy as umom
import csv 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler #and chocochip cake
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 



df = pd.read_csv("meandraytaarestilllovingeachotherunlikenikkisensei.csv")



factors = df[["variance", "skewness", "curtosis", "entropy"]]
rayta = df["class"]

factors_train, factors_test, rayta_train, rayta_test = train_test_split(factors, 
rayta, test_size = 0.25, random_state = 42)

#scaling units

MyBbRayta = LogisticRegression()
MyBbRayta.fit(factors_train,rayta_train)
raytaPred = MyBbRayta.predict(factors_test)

predicted_values = []
for i in raytaPred:
  if i == 0:
    predicted_values.append("Authorised")
  else:
    predicted_values.append("Forged")

actual_values = []
for i in rayta_test:
  if i == 0:
    actual_values.append("Authorised")
  else:
    actual_values.append("Forged")

labels = ["Forged","Authorised"]
#matrix form
cm = confusion_matrix(actual_values, predicted_values)

ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax)


ax.set_xlabel('Predicted')
ax.set_ylabel('Actual') 
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(labels); 
ax.yaxis.set_ticklabels(labels)
#accuracy

TrueNeg,FalsePos,FalseNeg,TruePos = confusion_matrix(rayta_test, raytaPred).ravel()

print("The true niggas are", TrueNeg)
print("The False positives are", FalsePos)
print("The false niggas are", FalseNeg)
print("the true positives are", TruePos)

accuracy = (TrueNeg + TruePos)*100/(TruePos+TrueNeg+FalsePos+FalseNeg)
print("Im gonna sing stay and yes and the accuracy yes yes ", accuracy)

