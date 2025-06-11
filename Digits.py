from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import matplotlib
import platform

# 한글 폰트 설정 (운영체제별)
if platform.system() == 'Windows':
    matplotlib.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':
    matplotlib.rc('font', family='AppleGothic')
else:
    matplotlib.rc('font', family='NanumGothic')

# 마이너스 깨짐 방지
matplotlib.rcParams['axes.unicode_minus'] = False

# 데이터 불러오기
digits = load_digits()
X = digits.data
y = digits.target

print("X의 shape:", X.shape)
print("첫 번째 샘플 벡터 (X[0]):\n", X[0])

# 훈련/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습 (SVM)
clf = SVC(gamma=0.001)
clf.fit(X_train, y_train)

# 예측
y_pred = clf.predict(X_test)

# 평가
print("정확도:", accuracy_score(y_test, y_pred))
print("분류 리포트:")
print(classification_report(y_test, y_pred))

# 시각화 (예측 결과 5개)
for i in range(5):
    plt.imshow(digits.images[i], cmap='gray')
    plt.title(f"예측: {clf.predict([X[i]])[0]}")
    plt.axis('off')
    plt.show()
