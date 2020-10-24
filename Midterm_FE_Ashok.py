import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from plotly import figure_factory as ff
from scipy import stats

import Assignment_4_FE
from Assignment_4_FE import CheckPredictors, CheckResponse


def fill_na(data):
    if isinstance(data, pd.Series):
        return data.fillna(0)
    else:
        return np.array([value if value is not None else 0 for value in data])


def cat_correlation(x, y, bias_correction=True, tschuprow=False):
    corr_coeff = np.nan
    try:
        x, y = fill_na(x), fill_na(y)
        crosstab_matrix = pd.crosstab(x, y)
        n_observations = crosstab_matrix.sum().sum()

        yates_correct = True
        if bias_correction:
            if crosstab_matrix.shape == (2, 2):
                yates_correct = False

        chi2, _, _, _ = stats.chi2_contingency(
            crosstab_matrix, correction=yates_correct
        )
        phi2 = chi2 / n_observations

        # r and c are number of categories of x and y
        r, c = crosstab_matrix.shape
        if bias_correction:
            phi2_corrected = max(0, phi2 - ((r - 1) * (c - 1)) / (n_observations - 1))
            r_corrected = r - ((r - 1) ** 2) / (n_observations - 1)
            c_corrected = c - ((c - 1) ** 2) / (n_observations - 1)
            if tschuprow:
                corr_coeff = np.sqrt(
                    phi2_corrected / np.sqrt((r_corrected - 1) * (c_corrected - 1))
                )
                return corr_coeff
            corr_coeff = np.sqrt(
                phi2_corrected / min((r_corrected - 1), (c_corrected - 1))
            )
            return corr_coeff
        if tschuprow:
            corr_coeff = np.sqrt(phi2 / np.sqrt((r - 1) * (c - 1)))
            return corr_coeff
        corr_coeff = np.sqrt(phi2 / min((r - 1), (c - 1)))
        return corr_coeff
    except Exception as ex:
        print(ex)
        if tschuprow:
            warnings.warn("Error calculating Tschuprow's T", RuntimeWarning)
        else:
            warnings.warn("Error calculating Cramer's V", RuntimeWarning)
        return corr_coeff


def cat_cont_correlation_ratio(categories, values):
    f_cat, _ = pd.factorize(categories)
    cat_num = np.max(f_cat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = values[np.argwhere(f_cat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
    numerator = np.sum(
        np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2))
    )
    denominator = np.sum(np.power(np.subtract(values, y_total_avg), 2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator / denominator)
    return eta


def CreateCorrMatrix(vartype1, vartype2, corr_coeff):
    # creating the correlation matrix:
    z_text = np.around(
        corr_coeff, decimals=2
    )  # Only show rounded value (full value on hover)

    fig = ff.create_annotated_heatmap(
        z=corr_coeff,
        y=vartype1,
        x=vartype2,
        annotation_text=z_text,
        colorscale="RdBu",
        hoverinfo="all",
    )

    # Make text size smaller
    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].font.size = 8

    return fig


def main(input_df_filename, response):
    input_df = pd.read_csv(input_df_filename)

    # First lets call the Assignment 4 script and store its output in a separate variable for further manipulations
    Assignment_4_FE.main(input_df_filename, response)
    Assignment4link = (
        "<a href=" + "/BDA696MLENG/Ashok_Assignment4.html" + ">" + "Assignment4link"
        "</a>"
    )
    # print(Assign4_result_df.head())

    # lets drop missing values so that no problems occur in the calculation
    input_df = input_df.dropna(axis=1, how="any")

    # lets drop the response variable and store the predictors in a separate df
    dropped_df = input_df.drop(response, axis=1)
    if CheckResponse(input_df, response) == "categorical":
        input_df[response] = input_df[response].astype("category")
        input_df[response] = input_df[response].cat.codes

    # lets start with splitting the predictors into categorical and continuous.
    cat = []
    cont = []
    for columns in dropped_df.columns:
        if CheckPredictors(input_df, columns) == "continuous":
            cont.append(columns)
        else:
            cat.append(columns)
    # these lists will be used for further computations.
    # lets start with creating the output dataframes.
    corr_df_cont_cont = pd.DataFrame(
        columns=[
            "Variable1",
            "Variable2",
            "Correlation coefficient",
            "p-value",
            "A4_link",
        ]
    )
    corr_df_cont_cat = pd.DataFrame(
        columns=["Variable1", "Variable2", "Correlation Coefficient", "A4_link"]
    )
    corr_df_cat_cat = pd.DataFrame(
        columns=["Variable1", "Variable2", "Correlation Coefficient", "A4_link"]
    )
    var1_cont_cont = []
    var2_cont_cont = []
    corr_cont_cont = []
    corr_coeff_cont_cont = []
    p = []
    # lets begin the correlation metrics for continuous continuous predictors:
    for cols in cont:
        temp = []
        for columns in cont:
            var1_cont_cont.append(cols)
            var2_cont_cont.append(columns)
            x = input_df[cols]
            y = input_df[columns]
            coeff, pv = stats.pearsonr(x, y)
            corr_cont_cont.append(coeff)
            temp.append(coeff)
            p.append(pv)
        corr_coeff_cont_cont.append(temp)
    corr_df_cont_cont["Variable1"] = var1_cont_cont
    corr_df_cont_cont["Variable2"] = var2_cont_cont
    corr_df_cont_cont["Correlation coefficient"] = corr_cont_cont
    corr_df_cont_cont["p-value"] = p
    corr_df_cont_cont["A4_link"] = Assignment4link
    # Lets order the dataframe in desc order of corr coeff
    corr_df_cont_cont.sort_values(
        by="Correlation coefficient", inplace=True, ascending=False
    )

    # # creating the correlation matrix:
    Fig_cont_cont = CreateCorrMatrix(cont, cont, corr_coeff_cont_cont)
    Fig_cont_cont.write_html(
        file=f"~/plots/Corr_Matrix_cont_cont.html",
        include_plotlyjs="cdn",
    )
    Cont_cont_corrmatrix = (
        "<a href ="
        + "~/plots/Corr_Matrix_cont_cont"
        + ".html"
        + ">"
        + "cont_cont_correlation_matrix"
        + "</a>"
    )
    corr_df_cont_cont["Corr Matrix Link"] = Cont_cont_corrmatrix
    # lets next calculate the correlation metrics for continuous categorical combinations
    corr_coeff_cont_cat = []
    var1_cont_cat = []
    var2_cont_cat = []
    corr_cont_cat = []
    for cols in cont:
        temp = []
        for columns in cat:
            var1_cont_cat.append(cols)
            var2_cont_cat.append(columns)
            x = cat_cont_correlation_ratio(input_df[columns], input_df[cols])
            corr_cont_cat.append(x)
            temp.append(x)
        corr_coeff_cont_cat.append(temp)
    corr_df_cont_cat["Variable1"] = var1_cont_cat
    corr_df_cont_cat["Variable2"] = var2_cont_cat
    corr_df_cont_cat["Correlation Coefficient"] = corr_cont_cat
    corr_df_cont_cat["A4_link"] = Assignment4link
    # lets order the df in descending order
    corr_df_cont_cat.sort_values(
        by="Correlation Coefficient", inplace=True, ascending=False
    )

    # creating the correlation matrix:
    Fig_cont_cat = CreateCorrMatrix(cont, cat, corr_coeff_cont_cat)
    Fig_cont_cat.write_html(
        file=f"~/plots/Corr_Matrix_cont_cat.html",
        include_plotlyjs="cdn",
    )
    Cont_cat_corrmatrix = (
        "<a href ="
        + "~/plots/Corr_Matrix_cont_cat"
        + ".html"
        + ">"
        + "cont_cat_correlation_matrix"
        + "</a>"
    )
    corr_df_cont_cat["CorrMatrix Link"] = Cont_cat_corrmatrix

    # lets next calculate the metrics for categorical categorical combinations
    corr_coeff_cat_cat = []
    var1_cat_cat = []
    var2_cat_cat = []
    corr_cat_cat = []
    for cols in cat:
        temp = []
        for columns in cat:
            var1_cat_cat.append(cols)
            var2_cat_cat.append(columns)
            x = cat_correlation(input_df[cols], input_df[columns])
            corr_cat_cat.append(x)
            temp.append(x)
        corr_coeff_cat_cat.append(temp)
    corr_df_cat_cat["Variable1"] = var1_cat_cat
    corr_df_cat_cat["Variable2"] = var2_cat_cat
    corr_df_cat_cat["Correlation Coefficient"] = corr_cat_cat
    corr_df_cat_cat["A4_link"] = Assignment4link
    # lets order the df in descending order
    corr_df_cat_cat.sort_values(
        by="Correlation Coefficient", inplace=True, ascending=False
    )

    # lets make the correlation matrix
    Fig_cat_cat = CreateCorrMatrix(cat, cat, corr_coeff_cat_cat)
    Fig_cat_cat.write_html(
        file=f"~/plots/Corr_Matrix_cat_cat.html",
        include_plotlyjs="cdn",
    )
    Cat_cat_corrmatrix = (
        "<a href ="
        + "~/plots/Corr_Matrix_cat_cat"
        + ".html"
        + ">"
        + "cat_cat_correlation_matrix"
        + "</a>"
    )
    corr_df_cat_cat["corr matrix link"] = Cat_cat_corrmatrix

    # Lets get started on the brute force variable combinations.
    # First we will do the continuous/continuous pairs.
    brute_force_cont_cont = pd.DataFrame(
        columns=[
            "Variable1",
            "Variable2",
            "DiffWMean",
            "DiffWMeanWeighted",
            "DWMHeatplot",
        ]
    )
    brute_force_cont_cat = pd.DataFrame(
        columns=[
            "Variable1",
            "Variable2",
            "DiffWMean",
            "DiffWMeanWeighted",
            "DWMHeatplot",
        ]
    )
    brute_force_cat_cat = pd.DataFrame(
        columns=[
            "Variable1",
            "Variable2",
            "DiffWmean",
            "DiffWMeanWeighted",
            "DWMHeatplot",
        ]
    )
    Var1_BF_cont_cont = []
    Var2_BF_cont_cont = []
    DiffMean_cont_cont = []
    DiffMeanW_cont_cont = []
    DWMHP_cont_cont = []
    for cols in cont:
        for columns in cont:
            if cols == columns:
                continue
            else:
                Var1_BF_cont_cont.append(cols)
                Var2_BF_cont_cont.append(columns)
                bin_df = pd.DataFrame(
                    {
                        "predval1": input_df[cols],
                        "predval2": input_df[columns],
                        "respval": input_df[response],
                        "bin1": pd.qcut(input_df[cols], 10, duplicates="drop"),
                        "bin2": pd.qcut(input_df[columns], 10, duplicates="drop"),
                    }
                )
                bin_df_group = bin_df.groupby(["bin1", "bin2"]).agg(
                    {"respval": ["count", "mean"]}
                )
                bin_df_group = bin_df_group.reset_index()
                bin_df_group.columns = [cols, columns, "BinCounts", "BinMeans"]
                bin_df_grouped = bin_df_group
                PopulationMean = input_df[response].mean()
                PopulationProportion = bin_df_grouped["BinCounts"] / sum(
                    bin_df_grouped["BinCounts"]
                )
                bin_df_grouped["MeanSqDiff"] = (
                    bin_df_grouped["BinMeans"] - PopulationMean
                ) ** 2
                bin_df_grouped["MeanSqDiffWeight"] = (
                    bin_df_grouped["MeanSqDiff"] * PopulationProportion
                )
                DiffMean_cont_cont.append(sum(bin_df_grouped["MeanSqDiff"]))
                DiffMeanW_cont_cont.append(sum(bin_df_grouped["MeanSqDiffWeight"]))
                heatmap_data = pd.pivot_table(
                    bin_df_grouped,
                    index=cols,
                    columns=columns,
                    values="MeanSqDiffWeight",
                )
                heatplot = sns.heatmap(heatmap_data, annot=True, cmap="RdBu")
                fig = heatplot.get_figure()
                fig.savefig("~/plots/BF" + cols + "" + columns + ".png")
                plt.clf()
                DWMHP_cont_cont.append(
                    "<a href ="
                    + "~/plots/BF"
                    + cols
                    + ""
                    + columns
                    + ".png"
                    + ">"
                    + "Heatplot"
                    + "</a>"
                )

    brute_force_cont_cont["Variable1"] = Var1_BF_cont_cont
    brute_force_cont_cont["Variable2"] = Var2_BF_cont_cont
    brute_force_cont_cont["DiffWMean"] = DiffMean_cont_cont
    brute_force_cont_cont["DiffWMeanWeighted"] = DiffMeanW_cont_cont
    brute_force_cont_cont["DWMHeatplot"] = DWMHP_cont_cont

    # next lets do the cont cat pairs:
    Var1_BF_cont_cat = []
    Var2_BF_cont_cat = []
    DiffMean_cont_cat = []
    DiffMeanW_cont_cat = []
    DWMHP_cont_cat = []
    for cols in cont:
        for columns in cat:
            Var1_BF_cont_cat.append(cols)
            Var2_BF_cont_cat.append(columns)
            bin_df = pd.DataFrame(
                {
                    "predval1": input_df[cols],
                    "predval2": input_df[columns],
                    "respval": input_df[response],
                    "bin1": pd.qcut(input_df[cols], 10, duplicates="drop"),
                }
            )
            bin_df_group = bin_df.groupby(["bin1", "predval2"]).agg(
                {"respval": ["count", "mean"]}
            )
            bin_df_group = bin_df_group.reset_index()
            bin_df_group.columns = [cols, columns, "BinCounts", "BinMeans"]
            bin_df_grouped = bin_df_group
            PopulationMean = input_df[response].mean()
            PopulationProportion = bin_df_grouped["BinCounts"] / sum(
                bin_df_grouped["BinCounts"]
            )
            bin_df_grouped["MeanSqDiff"] = (
                bin_df_grouped["BinMeans"] - PopulationMean
            ) ** 2
            bin_df_grouped["MeanSqDiffWeight"] = (
                bin_df_grouped["MeanSqDiff"] * PopulationProportion
            )
            DiffMean_cont_cat.append(sum(bin_df_grouped["MeanSqDiff"]))
            DiffMeanW_cont_cat.append(sum(bin_df_grouped["MeanSqDiffWeight"]))
            heatmap_data = pd.pivot_table(
                bin_df_grouped, index=cols, columns=columns, values="MeanSqDiffWeight"
            )
            heatplot = sns.heatmap(heatmap_data, annot=True, cmap="RdBu")
            fig = heatplot.get_figure()
            fig.savefig("~/plots/Output" + cols + "" + columns + ".png")
            plt.clf()
            DWMHP_cont_cat.append(
                "<a href ="
                + "~/plots/BF"
                + cols
                + ""
                + columns
                + ".png"
                + ">"
                + "Heatplot"
                + "</a>"
            )
    brute_force_cont_cat["Variable1"] = Var1_BF_cont_cat
    brute_force_cont_cat["Variable2"] = Var2_BF_cont_cat
    brute_force_cont_cat["DiffWMean"] = DiffMean_cont_cat
    brute_force_cont_cat["DiffWMeanWeighted"] = DiffMeanW_cont_cat
    brute_force_cont_cat["DWMHeatplot"] = DWMHP_cont_cat

    # lastly lets do the cat cat combinations
    Var1_BF_cat_cat = []
    Var2_BF_cat_cat = []
    DiffMean_cat_cat = []
    DiffMeanW_cat_cat = []
    DWMHP_cat_cat = []
    for cols in cat:
        for columns in cat:
            if cols == columns:
                continue
            else:
                Var1_BF_cat_cat.append(cols)
                Var2_BF_cat_cat.append(columns)
                bin_df = pd.DataFrame(
                    {
                        "predval1": input_df[cols],
                        "predval2": input_df[columns],
                        "respval": input_df[response],
                    }
                )
                bin_df_group = bin_df.groupby(["predval1", "predval2"]).agg(
                    {"respval": ["count", "mean"]}
                )
                bin_df_group = bin_df_group.reset_index()
                bin_df_group.columns = [cols, columns, "BinCounts", "BinMeans"]
                bin_df_grouped = bin_df_group
                PopulationMean = input_df[response].mean()
                PopulationProportion = bin_df_grouped["BinCounts"] / sum(
                    bin_df_grouped["BinCounts"]
                )
                bin_df_grouped["MeanSqDiff"] = (
                    bin_df_grouped["BinMeans"] - PopulationMean
                ) ** 2
                bin_df_grouped["MeanSqDiffWeight"] = (
                    bin_df_grouped["MeanSqDiff"] * PopulationProportion
                )
                DiffMean_cat_cat.append(sum(bin_df_grouped["MeanSqDiff"]))
                DiffMeanW_cat_cat.append(sum(bin_df_grouped["MeanSqDiffWeight"]))
                heatmap_data = pd.pivot_table(
                    bin_df_grouped,
                    index=cols,
                    columns=columns,
                    values="MeanSqDiffWeight",
                )
                heatplot = sns.heatmap(heatmap_data, annot=True, cmap="RdBu")
                fig = heatplot.get_figure()
                fig.savefig("~/plots/Output" + cols + "" + columns + ".png")
                plt.clf()
                DWMHP_cat_cat.append(
                    "<a href ="
                    + "~/plots/BF"
                    + cols
                    + ""
                    + columns
                    + ".png"
                    + ">"
                    + "Heatplot"
                    + "</a>"
                )
    brute_force_cat_cat["Variable1"] = Var1_BF_cat_cat
    brute_force_cat_cat["Variable2"] = Var2_BF_cat_cat
    brute_force_cat_cat["DiffWMean"] = DiffMean_cat_cat
    brute_force_cat_cat["DiffWMeanWeighted"] = DiffMeanW_cat_cat
    brute_force_cat_cat["DWMHeatplot"] = DWMHP_cat_cat

    # Lets put everything together and make the final 3 output htmls!
    # 1st will be the html with the correlation tables
    path = open("./Midterm_Ashok_Cor_tables.html", "w")
    path.write(
        corr_df_cont_cont.to_html(render_links=True, escape=False)
        + "<br>"
        + corr_df_cont_cat.to_html(render_links=True, escape=False)
        + "<br>"
        + corr_df_cat_cat.to_html(render_links=True, escape=False)
    )

    # finally we will make the brute force html
    path = open("./Midterm_Ashok_Brute_Force_tables.html", "w")
    path.write(
        brute_force_cont_cont.to_html(render_links=True, escape=False)
        + "<br>"
        + brute_force_cont_cat.to_html(render_links=True, escape=False)
        + "<br>"
        + brute_force_cat_cat.to_html(render_links=True, escape=False)
    )


if __name__ == "__main__":
    input_df_filename = sys.argv[1]
    response = sys.argv[2]
    sys.exit(main(input_df_filename, response))
