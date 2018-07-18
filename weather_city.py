import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC


def main():
    labelled_data = pd.read_csv(sys.argv[1])
    unlabelled_data = pd.read_csv(sys.argv[2])
    #print (labelled_data)
    X = labelled_data.drop(columns=['city','year']).values
    y = labelled_data['city'].values
    #print(X)
    #print(y)
    unlabelled_information = unlabelled_data.drop(columns=['city','year']).values


    # Training and testing the data
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    #GaussianNB model
    bayes_labelled_model = make_pipeline(
        StandardScaler(),
        GaussianNB(priors=None)
    )
    bayes_labelled_model.fit(X_train, y_train)
    #y_predicted = bayes_labelled_model.predict(X_test)
    #print('bayes_labelled_model accuracy_score: ', accuracy_score(y_test, y_predicted))
    #print('bayes_labelled_model score: ', bayes_labelled_model.score(X_test, y_test))

    #k-nearest neighbours classifiers
    knn_labelled_model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=6)
    )
    knn_labelled_model.fit(X_train, y_train)
    #y_predicted = knn_labelled_model.predict(X_test)
    #print('knn_labelled_model accuracy_score: ', accuracy_score(y_test, y_predicted))
    #print('knn_labelled_model score: ', knn_labelled_model.score(X_test, y_test))



    #svc
    svc_labelled_model = make_pipeline(
        StandardScaler(),
        SVC(kernel='linear', C=4)
    )
    svc_labelled_model.fit(X_train, y_train)
    #y_predicted = svc_labelled_model.predict(X_test)
    #print('svc_labelled_model accuracy_score: ', accuracy_score(y_test, y_predicted))
    print('SVC model score: ', svc_labelled_model.score(X_test, y_test))

    predictions = svc_labelled_model.predict(unlabelled_information)
    #print(predictions)
    #predictions1 = knn_labelled_model.predict(unlabelled_information)
    #print(predictions1)
    #predictions2 = bayes_labelled_model.predict(unlabelled_information)
    #print(predictions2)

    pd.Series(predictions).to_csv(sys.argv[3], index=False)

    df = pd.DataFrame({'truth': y_test, 'prediction': svc_labelled_model.predict(X_test)})
    #print(df[df['truth'] != df['prediction']])



if __name__ == '__main__':
    main()
