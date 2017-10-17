---
title: HW-3
date: 2017-10-14 16:10:42
tags: HW
---

## Classification of diabetes in Pima indian dataset
```python
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense,Activation
from sklearn.model_selection import train_test_split

pima = pd.read_csv("diabetes.csv")

dataset = np.array(pima);
X = dataset[:,:-1];
Y = dataset[:,-1];

seed = 7
acclist = []
final_result = []

#Model defined
model = Sequential()
model.add(Dense(8, input_dim=8,activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

for i in range(10):
    x_train,x_test,y_train, y_test = train_test_split(X,Y,test_size=0.33,random_state=seed)
    #complile model
    model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
    # fit the model
    model.fit(x_train, y_train,validation_data=(x_test,y_test), epochs=150, batch_size=10)
    scores = model.evaluate(x_test,y_test)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    y_out=model.predict(x_test);

    for k in range(y_out.shape[0]):
        y_out[k]=1 if y_out[k] > 0.5 else 0
                                                           
    count=0

    for k in range(y_out.shape[0]):
        if (y_test[k]==1 and y_out[k]==1) or (y_test[k]==0 and y_out[k]==0):
            count += 1
        accuracy = count / y_out.shape[0] * 100
#print ("accuracy =", accuracy)
    acclist.append(accuracy);
#np.mean(acclist);
#final_result.append(acclist);
with open('relu3.txt','w') as output:
    output.write(str(acclist))
```
### Experimental Environment

**# of layers: 3 then added a layer below**
```
model.add(Dense(4, activation='relu'))
```

### Experimental Results
|# of epochs|# of layers|Activ. func.|Acc. (Avrg.)|Acc. (Max.)|
|---|---|---|---|---|
|150|2|ReLU|75.20|76.37|
|150|3|ReLU|70.08|71.65|
|200|2|ReLU|73.22|75.19|
|200|3|ReLU|67.72|69.29|

