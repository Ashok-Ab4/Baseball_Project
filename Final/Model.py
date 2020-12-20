# train test split
import sys

import pandas as pd
from sklearn import svm
from plotly import graph_objs as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,multilabel_confusion_matrix,roc_curve, roc_auc_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from FinalBDA696.scripts import Midterm_FE_Ashok as Corr_Brut_Plots, Assignment_4_FE as PredRespPlots


def main(input_df_name,response):
    input_df = pd.read_csv(input_df_name)
    input_df = input_df.dropna(axis=0, how="any")
    print(input_df.columns)


    #Upon Iteration, we found that HT_Streak and AT_Streak were nostradamus variables and were causing the model
    #performance to skew into unrealistic numbers. So we will drop them from the dataset.

    input_df = input_df.drop(['HT_Streak','AT_Streak'],axis=1)

    X = input_df.drop(['Home_team_wins'], axis=1)
    y = input_df["Home_team_wins"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.15, random_state=1)

    name = []
    Accuracy_fin = []
    Precision_fin = []
    Recall_fin = []
    F_Score = []
    AUC = []
    AUC_Plot = []
    # Making a Pipeline for Normalizing the data and fitting RandomForestClassifier
    pipeline = Pipeline(
        [("Normalize", Normalizer()), ("rf", RandomForestClassifier(random_state=1))]
    )

    pipeline.fit(X_train, y_train)
    predict = pipeline.predict(X_test)
    print(predict)

    # Accuracy of the RandomForestClassifier
    accuracy = accuracy_score(y_test, predict)
    name.append("RF")
    Accuracy_fin.append(accuracy)
    prec,rec,fsc,_ = precision_recall_fscore_support(y_test,predict,average="macro")
    Precision_fin.append(prec)
    Recall_fin.append(rec)
    F_Score.append(fsc)
    print("Accuracy of RF:", accuracy)
    print("Precision Recall Fscore support of RF:", precision_recall_fscore_support(y_test,predict,average="macro"))
    confusion_matrix = multilabel_confusion_matrix(y_test,predict)
    print("RF CM",confusion_matrix)
    roc_auc_rf = roc_auc_score(y_test,predict)
    AUC.append(roc_auc_rf)
    print("RF ROC AUC Score",roc_auc_rf)
    fpr, tpr, _ = roc_curve(y_test, predict, drop_intermediate=False)
    fig = go.Figure(data=go.Scatter(x=fpr, y=tpr, name="Model"))
    fig = fig.add_trace(
        go.Scatter(
            x=[0.0, 1.0],
            y=[0.0, 1.0],
            line=dict(dash="dash"),
            mode="lines",
            showlegend=False,
        )
    )

    # Label the figure
    fig.update_layout(
        title=f"Receiver Operator Characteristic (AUC={round(roc_auc_rf, 6)})",
        xaxis_title="False Positive Rate (FPR)",
        yaxis_title="True Positive Rate (TPR)",
    )

    fig.write_html(
                    file=f"../plots/ROC_FOR_RF_MODEL.html",
                    include_plotlyjs="cdn",
                )
    AUC_Plot.append(
                    "<a href ="
                    + "./plots/ROC_FOR_RF_MODEL"
                    + ".html"
                    + ">"
                    + "ROC Curve RF"
                    + "</a>"
                )
    # Making a pipeline for Normalizing the data and fitting SVM Classifier
    pipeline = Pipeline(
            [("Normalize", Normalizer()), ("SVM", svm.SVC(kernel="rbf"))]
        )
    pipeline.fit(X_train, y_train)
    predict = pipeline.predict(X_test)


    # Accuracy score for the SVMClassifier
    accuracy = accuracy_score(y_test, predict)
    name.append("SVM")
    Accuracy_fin.append(accuracy)
    prec,rec,fsc,_ = precision_recall_fscore_support(y_test,predict,average="macro")
    Precision_fin.append(prec)
    Recall_fin.append(rec)
    F_Score.append(fsc)
    print("Accuracy of SVM with linear kernel:", accuracy)
    print("Accuracy of SVM:", accuracy)
    print("Precision Recall Fscore support of SVM:", precision_recall_fscore_support(y_test,predict,average="macro"))
    confusion_matrix = multilabel_confusion_matrix(y_test,predict)
    print("SVM CM",confusion_matrix)
    roc_auc_svm = roc_auc_score(y_test,predict)
    AUC.append(roc_auc_svm)
    print("SVM ROC AUC Score",roc_auc_svm)
    fpr1, tpr1, _ = roc_curve(y_test, predict, drop_intermediate=False)
    fig = go.Figure(data=go.Scatter(x=fpr1, y=tpr1, name="Model"))
    fig = fig.add_trace(
        go.Scatter(
            x=[0.0, 1.0],
            y=[0.0, 1.0],
            line=dict(dash="dash"),
            mode="lines",
            showlegend=False,
        )
    )

    # Label the figure
    fig.update_layout(
        title=f"Receiver Operator Characteristic (AUC={round(roc_auc_svm, 6)})",
        xaxis_title="False Positive Rate (FPR)",
        yaxis_title="True Positive Rate (TPR)",
    )
    fig.write_html(
                    file=f"../plots/ROC_FOR_SVM_MODEL.html",
                    include_plotlyjs="cdn",
                )
    AUC_Plot.append(
                    "<a href ="
                    + "./plots/ROC_FOR_SVM_MODEL"
                    + ".html"
                    + ">"
                    + "ROC Curve SVM"
                    + "</a>"
                )
    # Making a Pipeline for Normalizing the data and fitting AdaBoostClassifier
    pipeline = Pipeline(
        [("Normalize", Normalizer()), ("AdaBoost", AdaBoostClassifier(random_state=1,n_estimators=100))]
    )

    pipeline.fit(X_train, y_train)
    predict = pipeline.predict(X_test)
    print(predict)
    # Accuracy of the AdaBoostClassifier
    accuracy = accuracy_score(y_test, predict)
    name.append("ADABoost")
    Accuracy_fin.append(accuracy)
    prec,rec,fsc,_ = precision_recall_fscore_support(y_test,predict,average="macro")
    Precision_fin.append(prec)
    Recall_fin.append(rec)
    F_Score.append(fsc)
    print("Accuracy of adaboost:", accuracy)
    print("Precision Recall Fscore support of AdaBoost:", precision_recall_fscore_support(y_test,predict,average="macro"))
    confusion_matrix = multilabel_confusion_matrix(y_test,predict)
    print("AdaBoost CM",confusion_matrix)
    roc_auc_AB = roc_auc_score(y_test,predict)
    AUC.append(roc_auc_AB)
    print("AdaBoost ROC AUC Score",roc_auc_AB)
    fpr, tpr, _ = roc_curve(y_test, predict, drop_intermediate=False)
    fig = go.Figure(data=go.Scatter(x=fpr, y=tpr, name="Model"))
    fig = fig.add_trace(
        go.Scatter(
            x=[0.0, 1.0],
            y=[0.0, 1.0],
            line=dict(dash="dash"),
            mode="lines",
            showlegend=False,
        )
    )

    # Label the figure
    fig.update_layout(
        title=f"Receiver Operator Characteristic (AUC={round(roc_auc_AB, 6)})",
        xaxis_title="False Positive Rate (FPR)",
        yaxis_title="True Positive Rate (TPR)",
    )
    fig.write_html(
                    file=f"../plots/ROC_FOR_AB_MODEL.html",
                    include_plotlyjs="cdn",
                )
    AUC_Plot.append(
                    "<a href ="
                    + "./plots/ROC_FOR_AB_MODEL"
                    + ".html"
                    + ">"
                    + "ROC Curve AB"
                    + "</a>"
                )

    # Making a Pipeline for Normalizing the data and fitting logistic regression model
    pipeline = Pipeline(
        [("Normalize", Normalizer()), ("logit", LogisticRegression(random_state=1))]
    )

    pipeline.fit(X_train, y_train)
    predict = pipeline.predict(X_test)
    print(predict)

    # Accuracy of the log
    accuracy = accuracy_score(y_test, predict)
    name.append("LogisticRegression")
    Accuracy_fin.append(accuracy)
    prec,rec,fsc,_ = precision_recall_fscore_support(y_test,predict,average="macro")
    Precision_fin.append(prec)
    Recall_fin.append(rec)
    F_Score.append(fsc)
    print("Accuracy of log:", accuracy)
    print("Precision Recall Fscore support of log:", precision_recall_fscore_support(y_test,predict,average="macro"))
    confusion_matrix = multilabel_confusion_matrix(y_test,predict)
    print("log CM",confusion_matrix)
    roc_auc_log = roc_auc_score(y_test,predict)
    AUC.append(roc_auc_log)
    print("log ROC AUC Score",roc_auc_log)
    fpr, tpr, _ = roc_curve(y_test, predict, drop_intermediate=False)
    fig = go.Figure(data=go.Scatter(x=fpr, y=tpr, name="Model"))
    fig = fig.add_trace(
        go.Scatter(
            x=[0.0, 1.0],
            y=[0.0, 1.0],
            line=dict(dash="dash"),
            mode="lines",
            showlegend=False,
        )
    )

    # Label the figure
    fig.update_layout(
        title=f"Receiver Operator Characteristic (AUC={round(roc_auc_log, 6)})",
        xaxis_title="False Positive Rate (FPR)",
        yaxis_title="True Positive Rate (TPR)",
    )
    fig.write_html(
                    file=f"../plots/ROC_FOR_log_MODEL.html",
                    include_plotlyjs="cdn",
                )
    AUC_Plot.append(
                    "<a href ="
                    + "./plots/ROC_FOR_log_MODEL"
                    + ".html"
                    + ">"
                    + "ROC Curve LOG"
                    + "</a>"
                )

    #Putting it All Together

    PRP = PredRespPlots.main(input_df, response)

    Corr1,Corr2,Corr3,Brut1,Brut2,Brut3 = Corr_Brut_Plots.main('OutputTable.csv',response)

    PRP.to_html("Predictor_vs_Response_Plots.html", render_links=True, escape=False)


    path = open("./Correlation_tables.html", "w")
    path.write(
        Corr1.to_html(render_links=True, escape=False)
        + "<br>"
        + Corr2.to_html(render_links=True, escape=False)
        + "<br>"
        + Corr3.to_html(render_links=True, escape=False)
    )

    # finally we will make the brute force html
    path = open("./Brute_Force_tables.html", "w")
    path.write(
        Brut1.to_html(render_links=True, escape=False)
        + "<br>"
        + Brut2.to_html(render_links=True, escape=False)
        + "<br>"
        + Brut3.to_html(render_links=True, escape=False)
    )

    ModelResult_df = pd.DataFrame(
            columns=[
                "Name",
                "Accuracy",
                "Precision",
                "Recall",
                "F_score",
                "AUC",
                "AUC_Curve"
            ]
        )
    ModelResult_df["Name"] = name
    ModelResult_df["Accuracy"] = Accuracy_fin
    ModelResult_df["Precision"] = Precision_fin
    ModelResult_df["Recall"] = Recall_fin
    ModelResult_df["F_score"] = F_Score
    ModelResult_df["AUC"] = AUC
    ModelResult_df["AUC_Curve"] = AUC_Plot

    print(ModelResult_df.head())

    ModelResult_df.to_html("Model_Performance.html",render_links=True,escape=False)

if __name__ == "__main__":
    input_df_filename = "OutputTable.csv"  # sys.argv[1]
    response = "Home_team_wins"  # sys.argv[2]
    sys.exit(main(input_df_filename, response))
