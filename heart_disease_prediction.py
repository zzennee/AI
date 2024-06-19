
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 데이터 로드 및 전처리
data = pd.read_csv('heart.csv')
X = data.drop('target', axis=1)
y = data['target']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 초기 성능 평가
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Initial Accuracy: {accuracy}')

from sklearn.model_selection import cross_val_score

# k-폴드 교차 검증
cv_scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')
print(f'Cross-Validation Scores: {cv_scores}')
print(f'Mean CV Score: {cv_scores.mean()}')

import numpy as np
from sklearn.utils import resample

# 부트스트랩핑
n_iterations = 1000
n_size = int(len(X) * 0.50)
bootstrap_scores = []

for i in range(n_iterations):
    X_sample, y_sample = resample(X, y, n_samples=n_size)
    model.fit(X_sample, y_sample)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    bootstrap_scores.append(score)

print(f'Bootstrap Scores: {bootstrap_scores}')
print(f'Mean Bootstrap Score: {np.mean(bootstrap_scores)}')
