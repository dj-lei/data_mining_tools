from dmtools.preprocessing import *


class Check(object):

    def __init__(self):
        pass

    def summary(self, df):
        """
        数据摘要
        """
        print(df.info(null_counts=True))
        return df.describe(percentiles=[.1, .2, .3, .4, .5, .6, .7, .8, .9], include='all')

    def discover_df(self, df, path=None):
        """
        探索pandas字段数据格式
        """
        constraints = discover_df(df)

        if path is None:
            return json.loads(constraints.to_json())
        else:
            with open(path, 'w') as f:
                json.dump(json.loads(constraints.to_json()), f, indent=4, sort_keys=True)

    def verify_df(self, df, constraints_path, repair=False):
        """
        校验pandas字段数据格式
        """
        v = verify_df(df, constraints_path, repair=repair)

        print('Constraints passing: %d\n' % v.passes)
        print('Constraints failing: %d\n' % v.failures)
        print(str(v))
        print(v.to_frame())

    def detect_df(self, df, constraints_path):
        """
        检查pandas字段数据格式
        """
        v = detect_df(df, constraints_path)

        return v.to_dataframe()

    def pdextract(self, df, column):
        """
        提取正则表达式
        """
        return pdextract(df[column])


class Clean(object):

    def __init__(self):
        pass

    def standard_scaler(self, train, test):
        """
        特征标准化，均值0方差1的正态分布
        """
        scaler = preprocessing.StandardScaler().fit(train)
        return pd.DataFrame(scaler.transform(train), columns=train.columns), pd.DataFrame(scaler.transform(test), columns=test.columns)

    def min_max_scaler(self, train, test, feature_range=(0, 1)):
        """
        另一种标准化方法是将特征缩放到给定的最小值和最大值之间，通常介于零和一之间，或者将每个特征的最大绝对值缩放到单位大小.
        使用这种缩放的动机包括对特征的非常小的标准偏差具有鲁棒性，并且在稀疏数据中保留零条目。
        """
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=feature_range)
        X_train_minmax = min_max_scaler.fit_transform(train)
        return pd.DataFrame(X_train_minmax, columns=train.columns), pd.DataFrame(min_max_scaler.transform(test), columns=test.columns)

    def normalize(self, train, norm='l2'):
        """
        标准化是将单个样本缩放为具有单位范数的过程
        """
        normalizer = preprocessing.Normalizer(norm=norm).fit(train)  # fit does nothing / normalizer.transform(X)

        return normalizer

    def ordinal_encoder(self, train, columns):
        """
        分类特征连续编码
        """
        enc = preprocessing.OrdinalEncoder()
        enc.fit(train[columns])
        return enc

    def one_hot_encoder(self, train, columns):
        """
        one-hot分类特征编码
        """
        enc = preprocessing.OneHotEncoder()
        enc.fit(train[columns])
        return enc

    def kbins_discretizer(self, train, n_bins, encode, strategy):
        """
        bin离散

        Parameters
        ----------
        n_bins : int or array-like, shape (n_features,) (default=5)
            The number of bins to produce. Raises ValueError if ``n_bins < 2``.

        encode : {'onehot', 'onehot-dense', 'ordinal'}, (default='onehot')
            Method used to encode the transformed result.

            onehot
                Encode the transformed result with one-hot encoding
                and return a sparse matrix. Ignored features are always
                stacked to the right.
            onehot-dense
                Encode the transformed result with one-hot encoding
                and return a dense array. Ignored features are always
                stacked to the right.
            ordinal
                Return the bin identifier encoded as an integer value.

        strategy : {'uniform', 'quantile', 'kmeans'}, (default='quantile')
            Strategy used to define the widths of the bins.

            uniform
                All bins in each feature have identical widths.
            quantile
                All bins in each feature have the same number of points.
            kmeans
                Values in each bin have the same nearest center of a 1D k-means
        """
        est = preprocessing.KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy).fit(train)
        return est

    def binarizer_discretizer(self, threshold):
        """
        二进制离散

        Parameters
        ----------
        threshold : float, optional (0.0 by default)
            Feature values below or equal to this are replaced by 0, above it by 1.
            Threshold may not be less than 0 for operations on sparse matrices.

        copy : boolean, optional, default True
            set to False to perform inplace binarization and avoid a copy (if
            the input is already a numpy array or a scipy.sparse CSR matrix).
        """
        binarizer = preprocessing.Binarizer(threshold=threshold)
        return binarizer

    def custom_transformers(self):
        """
        自定义转换
        """
        pass

    def simple_imputer(self, train, missing_values, strategy, fill_value):
        """
        简单归因

        Parameters
        ----------
        missing_values : number, string, np.nan (default) or None
            The placeholder for the missing values. All occurrences of
            `missing_values` will be imputed.

        strategy : string, default='mean'
            The imputation strategy.

            - If "mean", then replace missing values using the mean along
              each column. Can only be used with numeric data.
            - If "median", then replace missing values using the median along
              each column. Can only be used with numeric data.
            - If "most_frequent", then replace missing using the most frequent
              value along each column. Can be used with strings or numeric data.
            - If "constant", then replace missing values with fill_value. Can be
              used with strings or numeric data.

            .. versionadded:: 0.20
               strategy="constant" for fixed value imputation.

        fill_value : string or numerical value, default=None
            When strategy == "constant", fill_value is used to replace all
            occurrences of missing_values.
            If left to the default, fill_value will be 0 when imputing numerical
            data and "missing_value" for strings or object data types.
        """
        imp = SimpleImputer(missing_values=missing_values, strategy=strategy, fill_value=fill_value)
        imp.fit(train)
        return imp

    def iterative_imputer(self):
        """
        Multivariate imputer that estimates each feature from all the others.

        A strategy for imputing missing values by modeling each feature with
        missing values as a function of other features in a round-robin fashion.
        """
        pass

    def knn_imputer(self, missing_values=np.nan, n_neighbors=5, weights="uniform"):
        """
        Imputation for completing missing values using k-Nearest Neighbors.
        Parameters
        ----------
        missing_values : number, string, np.nan or None, default=`np.nan`
            The placeholder for the missing values. All occurrences of
            `missing_values` will be imputed.

        n_neighbors : int, default=5
            Number of neighboring samples to use for imputation.

        weights : {'uniform', 'distance'} or callable, default='uniform'
            Weight function used in prediction.  Possible values:

            - 'uniform' : uniform weights. All points in each neighborhood are
              weighted equally.
            - 'distance' : weight points by the inverse of their distance.
              in this case, closer neighbors of a query point will have a
              greater influence than neighbors which are further away.
            - callable : a user-defined function which accepts an
              array of distances, and returns an array of the same shape
              containing the weights.

        metric : {'nan_euclidean'} or callable, default='nan_euclidean'
            Distance metric for searching neighbors. Possible values:

            - 'nan_euclidean'
            - callable : a user-defined function which conforms to the definition
              of ``_pairwise_callable(X, Y, metric, **kwds)``. The function
              accepts two arrays, X and Y, and a `missing_values` keyword in
              `kwds` and returns a scalar distance value.
        """
        imputer = KNNImputer(missing_values=missing_values, n_neighbors=n_neighbors, weights=weights)
        return imputer

    def isolation_forest(self, train, n_estimators=100, max_samples='auto', contamination='auto', max_features=1):
        """
        异常检测
        Return the anomaly score of each sample using the IsolationForest algorithm

        Parameters
        ----------
        n_estimators : int, optional (default=100)
            The number of base estimators in the ensemble.

        max_samples : int or float, optional (default="auto")
            The number of samples to draw from X to train each base estimator.
                - If int, then draw `max_samples` samples.
                - If float, then draw `max_samples * X.shape[0]` samples.
                - If "auto", then `max_samples=min(256, n_samples)`.

            If max_samples is larger than the number of samples provided,
            all samples will be used for all trees (no sampling).

        contamination : 'auto' or float, optional (default='auto')
            The amount of contamination of the data set, i.e. the proportion
            of outliers in the data set. Used when fitting to define the threshold
            on the scores of the samples.

                - If 'auto', the threshold is determined as in the
                  original paper.
                - If float, the contamination should be in the range [0, 0.5].

            .. versionchanged:: 0.22
               The default value of ``contamination`` changed from 0.1
               to ``'auto'``.

        max_features : int or float, optional (default=1.0)
            The number of features to draw from X to train each base estimator.

                - If int, then draw `max_features` features.
                - If float, then draw `max_features * X.shape[1]` features.
        """
        outlier_detection = IsolationForest(n_estimators=n_estimators, max_samples=max_samples, contamination=contamination, max_features=max_features)
        outlier_detection.fit(train)
        return outlier_detection

    def one_class_svm(self, train, kernel='rbf', gamma='scale', nu=0.5):
        """
        对异常值敏感，因此在异常值检测方面表现不佳。 当训练集不受异常值污染时，此估计器最适合新数据异常性检测

        Parameters
        ----------
        kernel : string, optional (default='rbf')
             Specifies the kernel type to be used in the algorithm.
             It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or
             a callable.
             If none is given, 'rbf' will be used. If a callable is given it is
             used to precompute the kernel matrix.

        degree : int, optional (default=3)
            Degree of the polynomial kernel function ('poly').
            Ignored by all other kernels.

        gamma : {'scale', 'auto'} or float, optional (default='scale')
            Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.

            - if ``gamma='scale'`` (default) is passed then it uses
              1 / (n_features * X.var()) as value of gamma,
            - if 'auto', uses 1 / n_features.

            .. versionchanged:: 0.22
               The default value of ``gamma`` changed from 'auto' to 'scale'.

        coef0 : float, optional (default=0.0)
            Independent term in kernel function.
            It is only significant in 'poly' and 'sigmoid'.

        tol : float, optional
            Tolerance for stopping criterion.

        nu : float, optional
            An upper bound on the fraction of training
            errors and a lower bound of the fraction of support
            vectors. Should be in the interval (0, 1]. By default 0.5
            will be taken.
        """
        inliers_detection = svm.OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
        inliers_detection.fit(train)
        return inliers_detection


class reduce(object):

    def __init__(self):
        pass

    def pca(self):
        """
        """
        pass