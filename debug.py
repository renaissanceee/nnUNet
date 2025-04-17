import pickle
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

# 生成模拟数据
X, y = make_classification(n_samples=10000, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练一个分类模型（以随机森林为例）
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train) # (7000, 20), (7000,)

# 获取训练集的预测概率（用于校准）
train_probs = model.predict_proba(X_train)[:, 1]  # 取正类概率

# 初始化并拟合 IsotonicRegression
isotonic = IsotonicRegression(out_of_bounds='clip')
isotonic.fit(train_probs, y_train)# (7000,), (7000,)

# 校准测试集概率
test_probs = model.predict_proba(X_test)[:, 1]
calibrated_probs = isotonic.transform(test_probs)

# 评估校准效果
print("原始概率的Brier分数:", brier_score_loss(y_test, test_probs))
print("校准后的Brier分数:", brier_score_loss(y_test, calibrated_probs))


# 保存整个校准器对象
with open('isotonic_calibrator.pkl', 'wb') as f:
    pickle.dump(isotonic, f)
# print("训练集概率范围:", (isotonic.X_min_, isotonic.X_max_))

########################################
print("start load pkl ...")
# 加载已保存的校准器
with open('isotonic_calibrator.pkl', 'rb') as f:
    isotonic = pickle.load(f)
import pdb;pdb.set_trace()
calibrated_probs = isotonic.transform(test_probs)
print("校准后的Brier分数:", brier_score_loss(y_test, calibrated_probs))