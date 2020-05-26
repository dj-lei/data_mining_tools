from dmtools.evaluate import *


class BaseEvaluate(object):

    def __init__(self):
        pass

    def mean_confidence_interval(self, data, confidence=0.95):
        """
        计算置信区间
        """
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
        return m, m-h, m+h


class ClassifierEvaluate(BaseEvaluate):

    def confusion_matrix(self, y_true, y_pred, desc):
        """
        混淆矩阵
        """
        print(desc)
        f, ax = plt.subplots(figsize=(15, 10))
        c2 = confusion_matrix(y_true, y_pred)
        sns.heatmap(c2, annot=True, ax=ax)

        ax.set_title('confusion matrix')
        ax.set_xlabel('predict')
        ax.set_ylabel('true')

    def roc_auc(self, clf, X_train, X_test, y_train, y_test, n_classes):
        """
        roc曲线与曲面积 / y = label_binarize(y, classes=list(range(n_classes)))
        基尼系数 gini = 2*auc - 1 / 其中AUC是曲线下的面积,基尼系数超过60％代表一个好的模型
        """
        # Learn to predict each class against the other
        classifier = OneVsRestClassifier(clf)
        y_score = classifier.fit(X_train, y_train).decision_function(X_test)

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure()

        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f}, gini = {2:0.2f})' ''.format(i, roc_auc[i], roc_auc[i]*2 -1))

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        plt.show()