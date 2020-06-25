""" PyBox Package: module: data_explorers.py
This is a module for investigating predictors by plotting correlations and distribtuions against target

It requires:
    The module configs.py for reading the config # TODO: currently the config is pretty hardcoded
It produces:
    * Explorer: Main object comprising data and provide plotting methods
It consists further following auxiliary classes:

"""

import numpy as np
import pandas as pd
from PyBox import configs
from .utils import generate_dtype_dict
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import seaborn as sns
from scipy.stats import chi2_contingency
from scipy.cluster.hierarchy import ward, fcluster
from sklearn.metrics import roc_auc_score
import pdb


class Explorer:

    def __init__(self, conf: configs.ExplorerFreqSevConfig, df: pd.DataFrame):
        self.conf = conf  # TODO: want a copy here?

        if df is not None:
            self.df_freq = df.copy()
        else:
            # TODO: maybe want "get" methods here instead of direct attribute getting?
            # TODO: or does it make sense to inherit from Config class like for "FreqSevConfig(Config)"  or directly form FreqSevConfig?
            # TODO: should "cn" be prefix or suffix: found both in other programs
            # TODO: search for every "self." or "self.conf"
            dtype_dict = generate_dtype_dict(predictors_metr=self.conf.cn_metrs,
                                             predictors_nomi=self.conf.cn_nomis)
            self.df_freq = pd.read_csv(conf.data_fn, sep='|', dtype=dtype_dict, nrows=100000)
            self.df_freq[self.conf.cn_nomis] = self.df_freq[self.conf.cn_nomis].fillna("(Missing)")
            self.df_freq[self.conf.cn_nomis] = (self.df_freq[self.conf.cn_nomis]
                                                .apply(lambda x: x.where(np.in1d(x, x.value_counts().index.values[:20]),
                                                                         other="_OTHER_")))

        self.df_sev = self.df_freq.loc[self.df_freq[self.conf.cn_target_freq] == self.conf.freq_pos_str, :]

        # Calc varimp
        self.varimp_freq = self.__calc_imp(df=self.df_freq, features=np.append(self.conf.cn_metrs, self.conf.cn_nomis),
                                           target=self.conf.cn_target_freq, target_type="freq")
        self.varimp_sev = self.__calc_imp(df=self.df_sev, features=np.append(self.conf.cn_metrs, self.conf.cn_nomis),
                                          target=self.conf.cn_target_sev, target_type="sev")
        features=np.append(self.conf.cn_metrs, self.conf.cn_nomis)
        self.varimp_freq = pd.Series(range(len(features)), index=features)
        self.varimp_sev = self.varimp_freq

    # Univariate variable importance
    # TODO: should we split further into 2 methods __calc_imp_freq/sev?
    # TODO: ... and these split again into __calc_imp_nomi_freq like for "__plot_single_nomi_freq"?
    def __calc_imp(self, df, features, target, target_type="freq"):
        varimp = pd.Series()
        for feature_act in features:
            # feature_act=cate[0]
            if target_type == "freq":
                print(feature_act)
                #pdb.set_trace()
                y_true = np.where(df[target] == self.conf.freq_pos_str, 1, 0)
                if df[feature_act].dtype == "object":
                    dummy = df[feature_act]
                else:
                    # Create bins from metric variable
                    if df[feature_act].nunique() == 1:
                        dummy = df[feature_act]
                    else:
                        dummy = pd.qcut(df[feature_act], 10, duplicates="drop").astype("object").fillna("(Missing)")
                y_score = pd.Series(y_true).groupby(dummy).transform("mean").values
                varimp_act = {feature_act: round(roc_auc_score(y_true, y_score), 3)}
            if target_type == "sev":
                y_true = df[target]
                if df[feature_act].dtype == "object":
                    y_score = df.groupby(feature_act)[target].transform("mean")
                else:
                    y_score = df[feature_act]
                varimp_act = {feature_act: (abs(pd.DataFrame({"y_true": y_true, "y_score": y_score})
                                                .corr(method="spearman")
                                                .values[0, 1]).round(3))}
            varimp = varimp.append(pd.Series(varimp_act))

        varimp.sort_values(ascending=False, inplace=True)
        return varimp

    def __plot_single_nomi_freq(self, cn_nomi, ax):
        # Prepare data
        #pdb.set_trace()
        df_hlp = pd.crosstab(self.df_freq[cn_nomi], self.df_freq[self.conf.cn_target_freq])
        df_plot = df_hlp.div(df_hlp.sum(axis=1), axis=0)
        df_plot["w"] = df_hlp.sum(axis=1)
        df_plot = df_plot.reset_index()
        df_plot["pct"] = 100 * df_plot["w"] / len(self.df_freq)
        df_plot["w"] = 0.9 * df_plot["w"] / max(df_plot["w"])
        df_plot[cn_nomi + "_new"] = (df_plot[cn_nomi] + " (" +
                                         (df_plot["pct"]).round(1).astype(str) + "%)")
        min_width = 0
        df_plot["new_w"] = np.where(df_plot["w"].values < min_width, min_width, df_plot["w"])

        # Main barplot
        ax.barh(df_plot[cn_nomi + "_new"], df_plot[self.conf.freq_pos_str], height=df_plot.new_w,
                    color="red", edgecolor="black", alpha=0.5, linewidth=1)
        ax.set_xlabel("mean(" + self.conf.cn_target_freq + ")")
        # ax_act.set_yticklabels(df_plot[feature_act + "_new"].values)
        # ax_act.set_yticklabels(df_plot[feature_act].values)
        ax.set_title(cn_nomi + " (VI: " + str(self.varimp_freq[cn_nomi]) + ")")
        ax.axvline(np.mean(np.where(self.df_freq[self.conf.cn_target_freq] == self.conf.freq_pos_str, 1, 0)),
                   ls="dotted", color="black")  # priori line

        # Inner barplot
        xlim = ax.get_xlim()
        ax.set_xlim(xlim[0] - 0.3 * (xlim[1] - xlim[0]))
        inset_ax = ax.inset_axes([0, 0, 0.2, 1])
        inset_ax.set_axis_off()
        ax.axvline(0, color="black")  # separation line
        # inset_ax.set_yticklabels(df_plot[feature_act + "_new"].values)
        ax.get_shared_y_axes().join(ax, inset_ax)
        inset_ax.barh(df_plot[cn_nomi + "_new"], df_plot.w,
                      color="lightgrey", edgecolor="black", linewidth=1)

        return df_plot # for evaluation purpose

    def __plot_single_metr_freq(self, cn_metr, ax):
        # Main distribution plot (overlayed)
        members = np.sort(self.df_freq[self.conf.cn_target_freq].unique())
        color = ("blue", "red")
        for m, member in enumerate(members):
            sns.distplot(self.df_freq.loc[self.df_freq[self.conf.cn_target_freq] == member, cn_metr].dropna(),
                         color=color[m],
                         bins=20,
                         label=member,
                         ax=ax)
        ax.set_title(cn_metr + " (VI: " + str(self.varimp_freq[cn_metr]) + ")")
        ax.set_ylabel("density")
        ax.set_xlabel(cn_metr + " (NA: " +
                          str((self.df_freq[cn_metr].isnull().mean() * 100).round(1)) +
                          "%)")
        ax.legend(title=self.conf.cn_target_freq, loc="best")

        # Inner Boxplot
        ylim = ax.get_ylim()
        ax.set_ylim(ylim[0] - 0.3 * (ylim[1] - ylim[0]))
        inset_ax = ax.inset_axes([0, 0, 1, 0.2])
        inset_ax.set_axis_off()
        ax.get_shared_x_axes().join(ax, inset_ax)
        i_bool = self.df_freq[cn_metr].notnull()
        sns.boxplot(x=self.df_freq.loc[i_bool, cn_metr],
                    y=self.df_freq.loc[i_bool, self.conf.cn_target_freq].astype("category"),
                    showmeans=True,
                    meanprops={"marker": "x", "markerfacecolor": "black", "markeredgecolor": "black"},
                    palette=color,
                    ax=inset_ax)

    def __plot_single_metr_sev(self, cn_metr, ax):

        # Main Heatmap

        # Calc scale
        if self.conf.ylim is not None:
            ax.set_ylim((0, self.conf.ylim))
            ymin = 0
            ymax = self.conf.ylim
            xmin = self.df_sev[cn_metr].min()
            xmax = self.df_sev[cn_metr].max()
        else:
            ymin = ymax = xmin = xmax = None

        # Calc colormap
        tmp_cmap = mcolors.LinearSegmentedColormap.from_list("gr_bl_yl_rd",
                                                             [(0.5, 0.5, 0.5, 0), "blue", "yellow",
                                                              "red"])
        # Hexbin plot
        ax.set_facecolor('0.98')
        p = ax.hexbin(self.df_sev[cn_metr], self.df_sev[self.conf.cn_target_sev],
                      extent=None if self.conf.ylim is None else (xmin, xmax, ymin, ymax),
                      cmap=tmp_cmap)
        plt.colorbar(p, ax=ax)
        #pdb.set_trace()
        ax.set_title(cn_metr + " (VI: " + str(self.varimp_sev[cn_metr]) + ")")
        ax.set_ylabel(self.conf.cn_target_sev)
        ax.set_xlabel(cn_metr + " (NA: " +
                      str(self.df_sev[cn_metr].isnull().mean().round(3) * 100) +
                      "%)")
        ylim = ax.get_ylim()
        # ax_act.grid(False)
        ax.axhline(color="grey")

        # Add lowess regression line?
        #pdb.set_trace()
        if len(self.df_sev) < 500:
            sns.regplot(cn_metr, self.conf.cn_target_sev, self.df_sev, lowess=True, scatter=False, color="black", ax=ax)

        # Inner Histogram
        ax.set_ylim(ylim[0] - 0.4 * (ylim[1] - ylim[0]))
        inset_ax = ax.inset_axes([0, 0.07, 1, 0.2])
        inset_ax.set_axis_off()
        ax.get_shared_x_axes().join(ax, inset_ax)
        i_bool = self.df_sev[cn_metr].notnull()
        sns.distplot(self.df_sev[cn_metr].dropna(), bins=20, color="black", ax=inset_ax)
        ax.axhline(np.mean(self.df_sev[self.conf.cn_target_sev]), ls="dotted", color="black")

        # Inner-inner Boxplot
        inset_ax = ax.inset_axes([0, 0.01, 1, 0.05])
        inset_ax.set_axis_off()
        inset_ax.get_shared_x_axes().join(ax, inset_ax)
        sns.boxplot(x=self.df_sev.loc[i_bool, cn_metr], palette=["grey"], ax=inset_ax)
        ax.set_xlabel(cn_metr + " (NA: " +
                      str(self.df_sev[cn_metr].isnull().mean().round(3) * 100) +
                      "%)")  # set it again!

    def __plot_single_nomi_sev(self, cn_nomi, ax):
        # Prepare data (Same as for CLASS target)
        df_plot = pd.DataFrame({"h": self.df_sev.groupby(cn_nomi)[self.conf.cn_target_sev].mean(),
                                "w": self.df_sev.groupby(cn_nomi).size()}).reset_index()
        df_plot["pct"] = 100 * df_plot["w"] / len(self.df_sev)
        df_plot["w"] = 0.9 * df_plot["w"] / max(df_plot["w"])
        df_plot[cn_nomi + "_new"] = (df_plot[cn_nomi] + " (" +
                                    (df_plot["pct"]).round(1).astype(str) + "%)")
        df_plot["new_w"] = np.where(df_plot["w"].values < 0.2, 0.2, df_plot["w"])

        # Main grouped boxplot
        if self.conf.ylim is not None:
            ax.set_xlim((0, self.conf.ylim))
        bp = self.df_sev[[cn_nomi, self.conf.cn_target_sev]].boxplot(self.conf.cn_target_sev, cn_nomi, vert=False,
                                                                 widths=df_plot.w.values,
                                                                 showmeans=True,
                                                                 meanprops=dict(marker="x",
                                                                                markeredgecolor="black"),
                                                                 flierprops=dict(marker="."),
                                                                 return_type='dict',
                                                                 ax=ax)
        [[item.set_color('black') for item in bp[self.conf.cn_target_sev][key]] for key in bp[self.conf.cn_target_sev].keys()]
        #fig.suptitle("")
        ax.set_xlabel(self.conf.cn_target_sev)
        ax.set_yticklabels(df_plot[cn_nomi + "_new"].values)
        #pdb.set_trace()
        ax.set_title(cn_nomi + " (VI: " + str(self.varimp_sev[cn_nomi]) + ")")
        ax.axvline(np.mean(self.df_sev[self.conf.cn_target_sev]), ls="dotted", color="black")

        # Inner barplot
        xlim = ax.get_xlim()
        ax.set_xlim(xlim[0] - 0.3 * (xlim[1] - xlim[0]))
        inset_ax = ax.inset_axes([0, 0, 0.2, 1])
        inset_ax.set_axis_off()
        inset_ax.get_shared_y_axes().join(ax, inset_ax)
        inset_ax.barh(df_plot.index.values + 1, df_plot.w, color="lightgrey", edgecolor="black",
                      linewidth=1)

        return df_plot # for evaluation purpose

    def __plot_distr(self, target_type):
        sns.set(style="whitegrid")
        n_ppp = self.conf.ncol * self.conf.nrow  # plots per page

        # Get plot_list
        if target_type == "freq":
            plot_list = self.varimp_freq.index.values
        if target_type == "sev":
            plot_list = self.varimp_sev.index.values

        for i, cn_feature in enumerate(plot_list):
            # Start new subplot on new page
            if i % n_ppp == 0:
                fig, ax = plt.subplots(self.conf.nrow, self.conf.ncol)
                fig.set_size_inches(w=self.conf.ncol*6, h=self.conf.nrow*6)
                i_ax = 0

            # Catch single plot case
            if n_ppp == 1:
                ax_act = ax
            else:
                ax_act = ax.flat[i_ax]

            if cn_feature in self.conf.cn_metrs:
                if target_type == "freq":
                    self.__plot_single_metr_freq(cn_metr=cn_feature, ax=ax_act)
                elif target_type == "sev":
                    self.__plot_single_metr_sev(cn_metr=cn_feature, ax=ax_act)
            if cn_feature in self.conf.cn_nomis:
                if target_type == "freq":
                    self.__plot_single_nomi_freq(cn_nomi=cn_feature, ax=ax_act)
                elif target_type == "sev":
                    self.__plot_single_nomi_sev(cn_nomi=cn_feature, ax=ax_act)
                    fig.suptitle("")
            i_ax += 1
            if i_ax == n_ppp or i == len(plot_list) - 1:
                fig.tight_layout()

    # TODO: Should it be possible to pass a features_plot_list?
    def plot_distr_freq(self):
        self.__plot_distr(target_type="freq")

    def plot_distr_sev(self):
        self.__plot_distr(target_type="sev")

    # TODO: should "metr" and "nomi" be compressed to "features" with adding features_type=metr/nomi and a corr_type with as enum
    # TODO: upgrade matplotlib to 3.1.2 at least
    def _plot_corr(self, metr=None, nomi=None, metr_corr_type="spearman", nomi_corr_type="contingency", n_cluster=5, show_plots=True):

        # Dummy init
        df_corr = None

        # All categorical variables
        if nomi is not None:
            # Intialize matrix with zeros
            df_corr = pd.DataFrame(np.zeros([len(nomi), len(nomi)]), index=nomi, columns=nomi)

            for i in range(len(nomi)):
                #print("cate=", nomi[i])
                for j in range(i + 1, len(nomi)):
                    # i=1; j=2
                    tmp = pd.crosstab(self.df_freq[nomi[i]], self.df_freq[nomi[j]])
                    n = np.sum(tmp.values)
                    m = min(tmp.shape)
                    chi2 = chi2_contingency(tmp)[0]

                    # try:
                    if nomi_corr_type == "contingency":
                        df_corr.iloc[i, j] = np.sqrt(chi2 / (n + chi2)) * np.sqrt(m / (m - 1))
                    elif nomi_corr_type == "cramersv":
                        df_corr.iloc[i, j] = np.sqrt(chi2 / (n * (m - 1)))
                    else:
                        df_corr.iloc[i, j] = None
                    # except:
                    # df_corr.iloc[i, j] = None
                    df_corr.iloc[j, i] = df_corr.iloc[i, j]
            d_new_names = dict(zip(df_corr.columns.values,
                                   df_corr.columns.values + " (" +
                                   self.df_freq[df_corr.columns.values].nunique().astype("str").values + ")"))
            df_corr.rename(columns=d_new_names, index=d_new_names, inplace=True)

        # All metric variables
        if metr is not None:
            df_corr = abs(self.df_freq[metr].corr(method=metr_corr_type))
            d_new_names = dict(zip(df_corr.columns.values,
                                   df_corr.columns.values + " (NA: " +
                                   (self.df_freq[df_corr.columns.values].isnull().mean() * 100).round(1).astype(
                                       "str").values + "%)"))
            df_corr.rename(columns=d_new_names, index=d_new_names, inplace=True)

        # Filter out rows or cols below cutoff
        np.fill_diagonal(df_corr.values, 0) # remove diagonal before calc of max per column
        i_bool = (df_corr.max(axis=1) > self.conf.corr_cutoff).values
        df_corr = df_corr.loc[i_bool, i_bool]
        np.fill_diagonal(df_corr.values, 1) # reset diagonal again

        # Cluster df_corr
        new_order = df_corr.columns.values[
            fcluster(ward(1 - np.triu(df_corr)), n_cluster, criterion='maxclust').argsort()]
        df_corr = df_corr.loc[new_order, new_order]

        # Plot
        if show_plots:
            fig, ax = plt.subplots(1, 1)
            ax_act = ax
            sns.heatmap(df_corr, annot=True, fmt=".2f", cmap="Blues", ax=ax_act, xticklabels=True, yticklabels=True)
            ax_act.set_yticklabels(labels=ax_act.get_yticklabels(), rotation=0)
            ax_act.set_xticklabels(labels=ax_act.get_xticklabels(), rotation=90)
            if metr is not None:
                ax_act.set_title("Absolute " + metr_corr_type + "correlation (cutoff at " + str(self.conf.corr_cutoff) + ")")
            if nomi is not None:
                if nomi_corr_type == "contingency":
                    ax_act.set_title("Contingency coefficient (cutoff at " + str(self.conf.corr_cutoff) + ")")
                if nomi_corr_type == "cramersv":
                    ax_act.set_title("Cramer's V (cutoff at " + str(self.conf.corr_cutoff) + ")")
            fig.set_size_inches(w=df_corr.shape[0] * 3, h=df_corr.shape[0] * 3)
            #pdb.set_trace()
            #fig.tight_layout()
            plt.show()
        # new: return data frame for testing purpose
        return df_corr

    def plot_corr(self):
        self._plot_corr(metr=self.conf.cn_metrs)
        self._plot_corr(nomi=self.conf.cn_nomis)