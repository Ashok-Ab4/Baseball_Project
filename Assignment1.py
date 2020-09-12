#!/usr/bin/env python3
import sys

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer


def main():
    column_id = ["Sepal length", "Sepal width", "Petal length", "Petal width", "class"]
    iris_df = pd.read_csv("Iris.data", names=column_id)
    # let's display the first few rows of the df
    print(iris_df.head())

    # missing values
    print("No of missing values are", iris_df.isnull().sum)

    # Summary stats
    print("mean of data:", np.mean(iris_df))
    print("max:", np.max(iris_df))
    print("min:", np.min(iris_df))

    # lets convert iris data to a np array to calculate quantiles
    iris_array = np.array(iris_df)
    print("25% quantile:", np.quantile(iris_array[:, :-1], q=0.25, axis=0))
    print("50% quantile:", np.quantile(iris_array[:, :-1], q=0.50, axis=0))
    print("75% quantile:", np.quantile(iris_array[:, :-1], q=0.75, axis=0))

    # visualizations:
    # plot1
    plt1 = px.violin(
        iris_df,
        x="class",
        y="Petal length",
        color="class",
        hover_data=iris_df.columns,
        title="violin plot to show Petal lengths for different Species",
    )
    plt1.show()

    # plot2
    plt2 = px.box(
        iris_df,
        x="class",
        y="Sepal length",
        hover_data=iris_df.columns,
        color="class",
        title="Bar plot to show difference in sepal length for species",
    )
    plt2.show()

    # plot3
    plt3 = px.scatter_matrix(
        iris_df,
        dimensions=["Sepal width", "Sepal length", "Petal width", "Petal length"],
        color="class",
        symbol="class",
        title="Scatter matrix of iris data set",
        labels={col: col.replace("_", " ") for col in iris_df.columns},
    )  # remove underscore
    plt3.update_traces(diagonal_visible=False)
    plt3.show()

    # plot4
    plt4 = px.bar(
        iris_df,
        x="Sepal length",
        y="Sepal width",
        color="class",
        title="Distribution of Sepal length and Sepal width with respect to species",
    )
    plt4.show()

    # plot5
    plt5 = px.scatter_3d(
        iris_df,
        x="Petal length",
        y="Petal width",
        z="Sepal width",
        color="Sepal length",
        symbol="class",
    )
    plt5.show()

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        iris_df.iloc[:, :-1].values,
        iris_df["class"],
        test_size=0.15,
        random_state=1234,
    )
    # Making a Pipeline for Normalizing the data and fitting RandomForestClassifier
    pipeline = Pipeline(
        [("Normalize", Normalizer()), ("rf", RandomForestClassifier(random_state=1234))]
    )
    pipeline.fit(X_train, y_train)
    predict = pipeline.predict(X_test)

    # Accuracy of the RandomForestClassifier
    accuracy = accuracy_score(y_test, predict)
    print("Accuracy of RF:", accuracy)

    # Making a pipeline for Normalizing the data and fitting SVM Classifier
    pipeline = Pipeline(
        [("Normalize", Normalizer()), ("SVM", svm.SVC(kernel="linear"))]
    )
    pipeline.fit(X_train, y_train)
    predict = pipeline.predict(X_test)
    # Accuracy score for the SVMClassifier
    accuracy = accuracy_score(y_test, predict)
    print("Accuracy of SVM with linear kernel:", accuracy)


if __name__ == "__main__":
    sys.exit(main())
