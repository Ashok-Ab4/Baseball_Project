import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objs as go
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix


def CheckResponse(input_df, response):
    unique = input_df[response].unique()
    if len(unique) > 2:
        return "continuous"
    else:
        return "boolean"


def CheckPredictors(input_df, pred):
    if (
        input_df[pred].dtypes == "object"
        or input_df[pred].nunique() / input_df[pred].count() < 0.05
    ):
        return "categorical"
    else:
        return "continuous"


def main(input_df_filename, response):
    # creating input and output dataframes
    input_df = pd.read_csv(input_df_filename)

    Result_df = pd.DataFrame(
        columns=[
            "Predictor",
            "Variable type",
            "plot link",
            "p-value",
            "t-score",
            "DiffWMean",
            "DiffWMeanWeighted",
            "DiffMeanPlot",
            "RandForestImp",
        ]
    )

    # create individual lists that will contain data for each column. Will be merged into dataframe in the final step.
    vartype = []
    pltlnk = []
    p = []
    t = []
    PTplot = []
    # DWM = []
    # DWMW = []
    # DM_Plt = []

    # checking if response is boolean or continuous
    x = CheckResponse(input_df, response)

    # Dropping response from the main table and storing it in separate variable
    y = input_df[response].values
    y1 = input_df[response]  # will be used later for RF classifier
    # del input_df[response]

    # lets start first with setting predictor in Result to the remaining columns in input df
    Result_df["Predictor"] = input_df.columns

    # lets begin with generating the plots necessary for the given variables
    # first lets create a directory called plots to store the plots in, in case it hasn't been made already
    if not os.path.exists("~/plots"):
        os.makedirs("~/plots")

    # lets now start with the different response/predictor combinations
    for columns in input_df.columns:
        columns1 = columns.replace(" ", "_")
        columnsf = columns1.replace("/", "_")
        if x == "continuous":
            if CheckPredictors(input_df, columns) == "categorical":
                vartype.append("Categorical")
                # since it is a continuous response and a categorical predictor, we will generate a distribution plot
                group_labels = [input_df[columns].unique()]
                # Lets assume that all categorical variables are only boolean, and they have 2 possible unique values
                # Slicing the group labels list to only contain the first two labels
                group1Data = y[input_df[columns] == group_labels[0]]
                group2Data = y[input_df[columns] == group_labels[1]]
                hist_data = [group1Data.values, group2Data.values]
                # creating distribution plot with custom bin size
                fig_1 = ff.create_distplot(hist_data, group_labels, bin_size=0.2)
                fig_1.update_layout(
                    title="Continuous Response by Categorical Predictor",
                    xaxis_title="Response",
                    yaxis_title="Distribution",
                )
                fig_1.show()
                fig_1.write_html(
                    file="~/plots/cont_response_cat_predictor_dist_plot"
                    + columnsf
                    + ".html",
                    include_plotlyjs="cdn",
                )
                # syntax of url is < a href = "url" > link text </a>
                pltlnk.append(
                    "<a href ="
                    + "~/plots/cont_response_cat_predictor_dist_plot"
                    + columnsf
                    + ".html"
                    + ">"
                    + "plt for"
                    + columnsf
                    + "</a>"
                )
            else:
                # since both the response and predictor are now continuous, lets proceed with a plot for that iteration
                vartype.append("Continuous")
                # for continuous/continuous combination, we will generate a trend line
                x1 = input_df[columns].values()
                y1 = y
                fig = px.scatter(x=x1, y=y1, trendline="ols")
                fig.update_layout(
                    title="Continuous Response by Continuous Predictor",
                    xaxis_title="Predictor",
                    yaxis_title="Response",
                )
                fig.show()
                fig.write_html(
                    file="~/plots/cont_response_cont_predictor_scatter_plot"
                    + columnsf
                    + ".html",
                    include_plotlyjs="cdn",
                )
                # lets add the link to the pltlnk list
                pltlnk.append(
                    "<a href ="
                    + "~/plots/cont_response_cont_predictor_scatter_plot"
                    + columnsf
                    + ".html"
                    + ">"
                    + "plt for"
                    + columnsf
                    + "</a>"
                )
        else:
            if CheckPredictors(input_df, columns) == "categorical":
                vartype.append("Categorical")
                # since both response and predictor are now categorical, we will generate a heatmap
                conf_matrix = confusion_matrix(input_df[columns], y)

                fig = go.Figure(
                    data=go.Heatmap(z=conf_matrix, zmin=0, zmax=conf_matrix.max())
                )
                fig.update_layout(
                    title="Categorical Predictor by Categorical Response (without relationship)",
                    xaxis_title="Response",
                    yaxis_title="Predictor",
                )
                fig.show()
                fig.write_html(
                    file="~/plots/cat_response_cat_predictor_heat_map"
                    + columnsf
                    + ".html",
                    include_plotlyjs="cdn",
                )
                # lets add the link to the pltlnk list
                pltlnk.append(
                    "<a href ="
                    + "~/plots/cat_response_cat_predictor_heat_map"
                    + columnsf
                    + ".html"
                    + ">"
                    + "plt for"
                    + columnsf
                    + "</a>"
                )
            else:
                vartype.append("Continuous")
                # since response is categorical and predictor is continuous, we will generate a violin plot
                fig_2 = go.Figure()
                group_labels = np.unique(y)
                group0Data = input_df[input_df[response] == group_labels[0]]
                group1Data = input_df[input_df[response] == group_labels[1]]
                hist_data = [group0Data, group1Data]
                for curr_hist, curr_group in zip(hist_data, group_labels):
                    fig_2.add_trace(
                        go.Violin(
                            x=np.repeat(curr_group, len(input_df)),
                            y=curr_hist,
                            name=str(curr_group),
                            box_visible=True,
                            meanline_visible=True,
                        )
                    )
                fig_2.update_layout(
                    title="Continuous Predictor by Categorical Response",
                    xaxis_title="Response",
                    yaxis_title="Predictor",
                )
                fig_2.show()
                fig_2.write_html(
                    file="~/plots/cat_response_cont_predictor_violin_plot"
                    + columnsf
                    + ".html",
                    include_plotlyjs="cdn",
                )
                # lets add the link to the pltlnk list
                pltlnk.append(
                    "<a href ="
                    + "~/plots/cat_response_cont_predictor_violin_plot"
                    + columnsf
                    + ".html"
                    + ">"
                    + "plt for"
                    + columnsf
                    + "</a>"
                )

        # lets move on to the p score and t scores.
        if x == "continuous":
            # for continuous responses, we use linear regression
            predictor = sm.add_constant(input_df[columns])
            linear_regression_model = sm.OLS(y, predictor)
            linear_regression_model_fitted = linear_regression_model.fit()

            # Get the stats
            t_value = round(linear_regression_model_fitted.tvalues[1], 6)
            t.append(t_value)
            p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])
            p.append(p_value)
            # Plot the figure
            fig = px.scatter(x=input_df[columns], y=y, trendline="ols")
            fig.update_layout(
                title=f"Variable: {columns}: (t-value={t_value}) (p-value={p_value})",
                xaxis_title=f"Variable: {columns}",
                yaxis_title="y",
            )
            fig.show()
            fig.write_html(file=f"~/plots/var_{columns}.html", include_plotlyjs="cdn")
            PTplot.append(
                "<a href ="
                + "~/plots/rankingplots"
                + columnsf
                + ".html"
                + ">"
                + "plt for"
                + columnsf
                + "</a>"
            )
        else:
            # for categorical responses, we use logistic regression
            predictor = sm.add_constant(input_df[columns])
            logistic_regression_model = sm.Logit(y, predictor)
            logistic_regression_model_fitted = logistic_regression_model.fit()

            # getting the statistics
            t_value = round(logistic_regression_model_fitted.tvalues[1], 6)
            t.append(t_value)
            p_value = "{:.6e}".format(logistic_regression_model_fitted.pvalues[1])
            p.append(p_value)

            # Plot the figure
            fig = px.scatter(x=input_df[columns], y=y, trendline="ols")
            fig.update_layout(
                title=f"Variable: {columns}: (t-value={t_value}) (p-value={p_value})",
                xaxis_title=f"Variable: {columns}",
                yaxis_title="y",
            )
            fig.show()
            fig.write_html(file=f"~/plots/var_{columnsf}.html", include_plotlyjs="cdn")
            PTplot.append(
                "<a href ="
                + "~/plots/rankingplots"
                + columnsf
                + ".html"
                + ">"
                + "plt for"
                + columnsf
                + "</a>"
            )

        # Lets do the Random Forest feature importance.
        if x == "continuous":
            # For continuous response we will use an RF regressor
            RandImp = RandomForestRegressor(
                n_estimators=65, oob_score=True, random_state=4
            )
            RandImp.fit(input_df, y)
            imp = RandImp.feature_importances_
        else:
            # for categorical response we will use an RF classifier, but first we will need to code the categorical response
            y1 = y1.astype("category")
            y1 = y1.cat.codes
            RandImp = RandomForestClassifier(
                n_estimators=65, oob_score=True, random_state=4
            )
            RandImp.fit(input_df, y1)
            imp = RandImp.feature_importances_

        # Lets do the mean and weighted means next

        # Lets put everything together in the output dataframe and save it as an html file
    Result_df["Variable type"] = vartype
    Result_df["p-value"] = p
    Result_df["t-score"] = t
    Result_df["plot link"] = pltlnk
    Result_df["RandForestImp"] = imp

    Result_df.to_html("Ashok_Assignment4.html", render_links=True, escape=False)


if __name__ == "__main__":
    input_df_filename = sys.argv[1]
    response = sys.argv[2]
    sys.exit(main(input_df_filename, response))
