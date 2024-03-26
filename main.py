import pandas as pd
import streamlit as st
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import RocCurveDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score
from sklearn.linear_model import LogisticRegression
df=pd.read_csv("creditcard.csv")
legit=df.drop(columns="Class",axis=1)
fraud=df["Class"]
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)
x_train,x_test,y_train,y_test=train_test_split(legit,fraud,test_size=0.2)
st.title("Multi-Algorithm Credit Card Fraud Detection System")
a=st.selectbox("Algorithms:",["RandomForest","GradientBoosting","LogisticRegression","SGD"])
col1, col2, col3 = st.columns([1,1,1])
with col1:
    b=st.button("Metrics")
with col2:
    c=st.button("Confusion Matrix")
with col3:
    d=st.button("ROC Curve")
st.write("Enter the following features to check if the transaction is legitimate or fraudulent:")
input_df = st.text_input('Input All features')
input_df_lst = input_df.split(',')
submit = st.button("Submit")
if(a=="LogisticRegression"):
    rd = LogisticRegression(max_iter=10000)
    rd.fit(x_train, y_train)
    if submit:

        features = np.array(input_df_lst, dtype=np.float64)

        prediction = rd.predict(features.reshape(1, -1))

        if prediction[0] == 0:
            st.write("Legitimate transaction")
        else:
            st.write("Fraudulent transaction")
    if(b==True):
        st.empty()
        st.write('Accuracy of the ',a,'is',accuracy_score(y_test,rd.predict(x_test)))
        st.write('Precision of the ',a,' is ',precision_score(y_test,rd.predict(x_test)))
        st.write('Recall of the ', a, ' is ', recall_score(y_test, rd.predict(x_test)))
        st.write('F1_score of the ', a, ' is ', f1_score(y_test, rd.predict(x_test)))
    elif(c==True):
        y_preds=rd.predict(x_test)
        st.write(confusion_matrix(y_test,y_preds))
    elif(d==True):
        RocCurveDisplay.from_estimator(rd,x_test,y_test)
        st.set_option('deprecation.showPyplotGlobalUse',False)
        st.pyplot()
elif(a=="RandomForest"):
    clf=RandomForestClassifier(n_estimators=10,random_state=23,max_depth=100)
    clf.fit(x_train, y_train)

    if submit:

        features = np.array(input_df_lst, dtype=np.float64)

        prediction = clf.predict(features.reshape(1, -1))

        if prediction[0] == 0:
            st.write("Legitimate transaction")
        else:
            st.write("Fraudulent transaction")
    if (b == True):
        st.empty()
        st.write('Accuracy of the ', a, 'is', accuracy_score(y_test, clf.predict(x_test)))
        st.write('Precision of the ', a, ' is ', precision_score(y_test, clf.predict(x_test)))
        st.write('Recall of the ', a, ' is ', recall_score(y_test, clf.predict(x_test)))
        st.write('F1_score of the ', a, ' is ', f1_score(y_test, clf.predict(x_test)))
    elif (c == True):
        y_preds = clf.predict(x_test)
        st.write(confusion_matrix(y_test, y_preds))
    elif (d == True):
        RocCurveDisplay.from_estimator(clf, x_test, y_test)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
elif(a=="GradientBoosting"):
    xg=GradientBoostingClassifier(n_estimators=10,random_state=20,max_depth=100)
    xg.fit(x_train, y_train)
    if submit:

        features = np.array(input_df_lst, dtype=np.float64)

        prediction = xg.predict(features.reshape(1, -1))

        if prediction[0] == 0:
            st.write("Legitimate transaction")
        else:
            st.write("Fraudulent transaction")
    if (b == True):
        st.empty()
        st.write('Accuracy of the ', a, 'is', accuracy_score(y_test, xg.predict(x_test)))
        st.write('Precision of the ', a, ' is ', precision_score(y_test, xg.predict(x_test)))
        st.write('Recall of the ', a, ' is ', recall_score(y_test, xg.predict(x_test)))
        st.write('F1_score of the ', a, ' is ', f1_score(y_test, xg.predict(x_test)))
    elif (c == True):
        y_preds = xg.predict(x_test)
        st.write(confusion_matrix(y_test, y_preds))
    elif (d == True):
        RocCurveDisplay.from_estimator(xg, x_test, y_test)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
else:
    sg=SGDClassifier()
    sg.fit(x_train, y_train)
    if submit:

        features = np.array(input_df_lst, dtype=np.float64)

        prediction = sg.predict(features.reshape(1, -1))

        if prediction[0] == 0:
            st.write("Legitimate transaction")
        else:
            st.write("Fraudulent transaction")
    if (b == True):
        st.empty()
        st.write('Accuracy of the ', a, 'is', accuracy_score(y_test, sg.predict(x_test)))
        st.write('Precision of the ', a, ' is ', precision_score(y_test, sg.predict(x_test)))
        st.write('Recall of the ', a, ' is ', recall_score(y_test, sg.predict(x_test)))
        st.write('F1_score of the ', a, ' is ', f1_score(y_test, sg.predict(x_test)))
    elif (c == True):
        y_preds = sg.predict(x_test)
        st.write(confusion_matrix(y_test, y_preds))
    elif (d == True):
        RocCurveDisplay.from_estimator(sg, x_test, y_test)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()