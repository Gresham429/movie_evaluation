from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from data.data_set import MovieDataset

class RandomForest:
    def __init__(self, n_estimators=100, random_state=None):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        return accuracy, report

def random_forest_predict(df):
    # 将数据集拆分为训练集和测试集
    train_data, test_data = train_test_split(df, test_size=0.1)

    # 准备数据加载器
    train_dataset = MovieDataset(train_data)
    test_dataset = MovieDataset(test_data)

    features_train = [sample['features'] for sample in train_dataset]
    rating_train = [sample['rating'] for sample in train_dataset]

    features_test = [sample['features'] for sample in test_dataset]
    rating_test = [sample['rating'] for sample in test_dataset]

    # 创建并训练随机森林模型
    rf_model = RandomForest()
    rf_model.train(features_train, rating_train)

    # 评估模型性能
    accuracy, report = rf_model.evaluate(features_test, rating_test)
    
    print(f'准确率: {accuracy:.4f}')
    print('\n分类报告:')
    print(report)

    return accuracy