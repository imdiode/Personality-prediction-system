import pandas as pd
from numpy import *
from sklearn import linear_model
import sklearn.metrics as metrics
import matplotlib.pyplot as plt


data =pd.read_csv('train dataset.csv')
array = data.values

for i in range(len(array)):
	if array[i][0]=="Male":
		array[i][0]=1
	else:
		array[i][0]=0


df=pd.DataFrame(array)

maindf =df[[0,1,2,3,4,5,6]]
mainarray=maindf.values
print (mainarray)


temp=df[7]
train_y=temp.values

for i in range(len(train_y)):
	train_y[i] =str(train_y[i])



mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg',max_iter =1000)
mul_lr.fit(mainarray, train_y)

testdata =pd.read_csv('test dataset.csv')
test = testdata.values

for i in range(len(test)):
	if test[i][0]=="Male":
		test[i][0]=1
	else:
		test[i][0]=0


df1=pd.DataFrame(test)

testdf =df1[[0,1,2,3,4,5,6]]
testval=df1[[7]]
maintestarray=testdf.values
print(maintestarray)

y_pred = mul_lr.predict(maintestarray)
for i in range(len(y_pred)) :
	y_pred[i]=str((y_pred[i]))
DF = pd.DataFrame(y_pred,columns=['Predicted Personality'])
DF.index=DF.index+1
DF.index.names = ['Person No']
DF.to_csv("output.csv")
print(DF)

print(metrics.accuracy_score(testval, y_pred, normalize=True, sample_weight=None))


plt.scatter(testdf[[0]], y_pred, color='red', label='0')
plt.scatter(testdf[[1]], y_pred, color='blue', label='1')
plt.scatter(testdf[[2]], y_pred, color='green', label='2')
plt.scatter(testdf[[3]], y_pred, color='black', label='3')
plt.scatter(testdf[[4]], y_pred, color='yellow', label='4')
plt.scatter(testdf[[5]], y_pred, color='cyan', label='5')
plt.scatter(testdf[[6]], y_pred, color='orange', label='6')
plt.legend()
plt.show()