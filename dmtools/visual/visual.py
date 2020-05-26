from dmtools.visual import *


class BaseVisual(object):

    def __init__(self):
        pass

    def summary_statistics(self, df):
        """
        连续变量汇总统计
        """
        df.describe()

    def continuous_attribute_distribution(self, df, columns):
        """
        单变量连续属性值直方图
        """
        fig, axes = plt.subplots(nrows=1, ncols=len(columns), figsize=(20, 5))
        for i, column in enumerate(columns):
            sns.distplot(df[column], hist_kws={'edgecolor': 'black'}, ax=axes[i])

    def correlation_matrix(self, df):
        """
        变量相关性矩阵
        """
        corr = df.corr()
        corr.style.background_gradient(cmap='coolwarm')


class ClassifierVisual(BaseVisual):

    def category_pie(self, df, target):
        """
        类型分布饼状图
        """
        labels = []
        sizes = []
        categories = list(set(df[target]))
        for category in categories:
            size = len(list(df.loc[(df[target] == category), target]))
            percent = size / len(df)
            labels.append(category)
            sizes.append(size)
            print(category, percent)

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
        ax1.axis('equal')
        plt.tight_layout()
        plt.show()

    def category_continuous_attribute_box(self, df, columns, target):
        """
        对应类别相关变量箱线图
        """
        fig, axes = plt.subplots(nrows=1, ncols=len(columns), figsize=(20, 5))
        for i, column in enumerate(columns):
            sns.boxplot(x=target, y=column, data=df, ax=axes[i])

    def category_feature_roc(self, df, var, target):
        """
        单变量与目标变量roc曲线面积
        """
        print(var + '/' + target + ' AUC: ' + str(round(roc_auc_score(df[target], df[var]), 6)))

    def category_time_probability(self, df, time_column, target):
        """
        分类时间可能性
        """
        date = pd.DataFrame()
        date['class probability'] = df.groupby(time_column)[target].mean()
        date['frequency'] = df.groupby(time_column)[target].size()
        date.plot(secondary_y='frequency', figsize=(20, 10))

    def category_scatter_diagram(self, df, target):
        """
        散布图矩阵
        """
        g = sns.PairGrid(df, hue=target)
        g.map_diag(plt.hist)
        g.map_offdiag(plt.scatter)
        g.add_legend()