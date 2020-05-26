from dmtools.algorithm import *


class BaseModel(object):

    def __init__(self, **options):
        self.clf = []
        self.x_train = []
        self.y_train = []

        try:
            self.save_path = options["save_path"]
        except KeyError:
            self.save_path = None

        try:
            self.scoring = options["scoring"]
        except KeyError:
            self.scoring = None

        try:
            self.cv = options["cv"]
        except KeyError:
            self.cv = 5

    @abstractmethod
    def cross_val_score(self):
        return cross_val_score(self.clf, self.x_train, self.y_train, scoring=self.scoring, cv=self.cv)

    @abstractmethod
    def train(self):
        self.clf.fit(self.x_train, self.y_train)

    @abstractmethod
    def predict(self, y_test):
        return self.clf.predict(y_test)


class Svm(BaseModel):
    """
    支持向量机

    C： C-SVC的惩罚参数C,默认值是1.0. C越大，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增大，
        趋向于对训练集全分对的情况，这样对训练集测试时准确率很高，但泛化能力弱。C值小，对误分类的惩罚减小，
        允许容错，将他们当成噪声点，泛化能力较强。

    kernel ：核函数，默认是rbf，可以是‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
    　　0 – 线性：u’v
    　　1 – 多项式：(gamma*u’*v + coef0)^degree
    　　2 – RBF函数：exp(-gamma|u-v|^2)
    　　3 –sigmoid：tanh(gamma*u’*v + coef0)

    degree ：多项式poly函数的维度，默认是3，选择其他核函数时会被忽略。
    gamma ： ‘rbf’,‘poly’ 和‘sigmoid’的核函数参数。默认是’auto’，则会选择1/n_features
    coef0 ：核函数的常数项。对于‘poly’和 ‘sigmoid’有用。
    probability ：是否采用概率估计？.默认为False
    shrinking ：是否采用shrinking heuristic方法，默认为true
    tol ：停止训练的误差值大小，默认为1e-3
    cache_size ：核函数cache缓存大小，默认为200
    class_weight ：类别的权重，字典形式传递。设置第几类的参数C为weight*C(C-SVC中的C)
    verbose ：允许冗余输出
    max_iter ：最大迭代次数。-1为无限制。
    decision_function_shape ：‘ovo’, ‘ovr’ or None, default=None3
    random_state ：数据洗牌时的种子值，int值
    """
    def __init__(self, x_train, y_train, **options):
        super(Svm, self).__init__(**options)
        self.x_train = x_train
        self.y_train = y_train

        try:
            self.param_C = options['C']
        except KeyError:
            self.param_C = 1

        try:
            self.kernel = options['kernel']
        except KeyError:
            self.kernel = 'rbf'

        try:
            self.max_iter = options['max_iter']
        except KeyError:
            self.max_iter = -1

        self.clf = svm.SVC(C=self.param_C, kernel=self.kernel, max_iter=self.max_iter)


class Knn(BaseModel):
    """
    最邻近

    n_neighbors ：邻近点个数，默认5。
    leaf_size：一般默认是30。可以在其值不大的范围内调试看看效果。
    weights ：参数有‘uniform’和‘distance’，可以选择调试。
    """

    def __init__(self, x_train, y_train, **options):
        super(Knn, self).__init__(**options)
        self.x_train = x_train
        self.y_train = y_train

        try:
            self.n_neighbors = options['n_neighbors']
        except KeyError:
            self.n_neighbors = 5

        try:
            self.weights = options['weights']
        except KeyError:
            self.weights = 'uniform'

        self.clf = neighbors.KNeighborsClassifier(n_neighbors=self.n_neighbors, weights=self.weights)


class GNB(BaseModel):
    """
    高斯朴素贝叶斯

    priors : array-like, shape (n_classes,)
        Prior probabilities of the classes. If specified the priors are not
        adjusted according to the data.

    var_smoothing : float, optional (default=1e-9)
        Portion of the largest variance of all features that is added to
        variances for calculation stability.
    """

    def __init__(self, x_train, y_train, **options):
        super(GNB, self).__init__(**options)
        self.x_train = x_train
        self.y_train = y_train

        try:
            self.priors = options['priors']
        except KeyError:
            self.priors = None

        try:
            self.var_smoothing = options['var_smoothing']
        except KeyError:
            self.var_smoothing = 1e-9

        self.clf = GaussianNB(priors=self.priors, var_smoothing=self.var_smoothing)


class RandomForest(BaseModel):
    """
    随机森林

    n_estimators=100 : 决策树的个数
    criterion="gini" : gini or entropy(default=”gini”)是计算属性的gini(基尼不纯度)还是entropy(信息增益)，来选择最合适的节点。
    max_depth=None : 设置树的最大深度，默认为None 容易过拟合
    min_samples_split=2 : 根据属性划分节点时，每个划分最少的样本数
    min_samples_leaf=1 : 叶子节点最少的样本数
    min_weight_fraction_leaf=0. : 叶子节点所需要的最小权值
    max_features="auto" : 选择最适属性时划分的特征不能超过此值
    max_leaf_nodes=None :  最大叶节点数
    min_impurity_decrease=0. : 不纯性的减少大于或等于此值，则拆分
    min_impurity_split=None : 一个节点的不存性大于此阈值则拆分，否则为叶节点
    bootstrap=True : 是否有放回的采样
    oob_score=False : 在某次决策树训练中没有被bootstrap选中的数据。多单个模型的参数训练，我们知道可以用cross validation（cv）来进行，
                      但是特别消耗时间，而且对于随机森林这种情况也没有大的必要，所以就用这个数据对决策树模型进行验证，算是一个简单的
                      交叉验证。性能消耗小，但是效果不错
    n_jobs=None : 并行job个数,CPU有多少core，就可以启动多少job
    random_state=None :
    verbose=0 :
    warm_start=False : 热启动，决定是否使用上次调用该类的结果然后增加新的
    class_weight=None : 各个label的权重
    ccp_alpha=0.0,
    max_samples=None : 如果bootstrap是true，每一次迭代采样数
    """

    def __init__(self, x_train, y_train, **options):
        super(RandomForest, self).__init__(**options)
        self.x_train = x_train
        self.y_train = y_train

        try:
            self.n_estimators = options['n_estimators']
        except KeyError:
            self.n_estimators = 100

        try:
            self.max_depth = options['max_depth']
        except KeyError:
            self.max_depth = 6

        try:
            self.min_samples_split = options['min_samples_split']
        except KeyError:
            self.min_samples_split = 2

        try:
            self.min_samples_leaf = options['min_samples_leaf']
        except KeyError:
            self.min_samples_leaf = 1

        try:
            self.max_features = options['max_features']
        except KeyError:
            self.max_features = None

        try:
            self.random_state = options['random_state']
        except KeyError:
            self.random_state = None

        self.clf = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                          min_samples_leaf=self.min_samples_leaf, max_features=self.max_features, random_state=self.random_state)


class Xgb(BaseModel):
    """
    Xgboost

    max_depth : int
        基本学习器树深.
    learning_rate : float
        学习率
    n_estimators : int
        Number of trees to fit.
    verbosity : int
        The degree of verbosity. Valid values are 0 (silent) - 3 (debug).
    silent : boolean
        Whether to print messages while running boosting. Deprecated. Use verbosity instead.
    objective : string or callable
        Specify the learning task and the corresponding learning objective. The objective options are below:
        reg:squarederror: regression with squared loss.
        reg:squaredlogerror: regression with squared log loss 12[𝑙𝑜𝑔(𝑝𝑟𝑒𝑑+1)−𝑙𝑜𝑔(𝑙𝑎𝑏𝑒𝑙+1)]2
            . All input labels are required to be greater than -1. Also, see metric rmsle for possible issue with this objective.
        reg:logistic: logistic regression
        binary:logistic: logistic regression for binary classification, output probability
        binary:logitraw: logistic regression for binary classification, output score before logistic transformation
        binary:hinge: hinge loss for binary classification. This makes predictions of 0 or 1, rather than producing probabilities.
        count:poisson –poisson regression for count data, output mean of poisson distribution
            max_delta_step is set to 0.7 by default in poisson regression (used to safeguard optimization)
        survival:cox: Cox regression for right censored survival time data (negative values are considered right censored). Note that predictions are returned on the hazard ratio scale (i.e., as HR = exp(marginal_prediction) in the proportional hazard function h(t) = h0(t) * HR).
        multi:softmax: set XGBoost to do multiclass classification using the softmax objective, you also need to set num_class(number of classes)
        multi:softprob: same as softmax, but output a vector of ndata * nclass, which can be further reshaped to ndata * nclass matrix. The result contains predicted probability of each data point belonging to each class.
        rank:pairwise: Use LambdaMART to perform pairwise ranking where the pairwise loss is minimized
        rank:ndcg: Use LambdaMART to perform list-wise ranking where Normalized Discounted Cumulative Gain (NDCG) is maximized
        rank:map: Use LambdaMART to perform list-wise ranking where Mean Average Precision (MAP) is maximized
        reg:gamma: gamma regression with log-link. Output is a mean of gamma distribution. It might be useful, e.g., for modeling insurance claims severity, or for any outcome that might be gamma-distributed.
        reg:tweedie: Tweedie regression with log-link. It might be useful, e.g., for modeling total loss in insurance, or for any outcome that might be Tweedie-distributed.
    booster: string
        Specify which booster to use: gbtree, gblinear or dart. default= gbtree
    n_jobs : int
        Number of parallel threads used to run xgboost.  (replaces ``nthread``)
    gamma : float
        Minimum loss reduction required to make a further partition on a leaf node of the tree.
    min_child_weight : int
        Minimum sum of instance weight(hessian) needed in a child.
    max_delta_step : int
        Maximum delta step we allow each tree's weight estimation to be.
    subsample : float
        Subsample ratio of the training instance.
    colsample_bytree : float
        Subsample ratio of columns when constructing each tree.
    colsample_bylevel : float
        Subsample ratio of columns for each level.
    colsample_bynode : float
        Subsample ratio of columns for each split.
    reg_alpha : float (xgb's alpha)
        L1 regularization term on weights
    reg_lambda : float (xgb's lambda)
        L2 regularization term on weights
    """

    def __init__(self, x_train, y_train, **options):
        super(Xgb, self).__init__(**options)
        self.x_train = x_train
        self.y_train = y_train

        try:
            self.n_estimators = options['n_estimators']
        except KeyError:
            self.n_estimators = 100

        try:
            self.max_depth = options['max_depth']
        except KeyError:
            self.max_depth = 6

        try:
            self.objective = options['objective']
        except KeyError:
            self.objective = "multi:softmax"

        try:
            self.random_state = options['random_state']
        except KeyError:
            self.random_state = 0

        self.clf = xgb.XGBClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, objective=self.objective, random_state=self.random_state)
