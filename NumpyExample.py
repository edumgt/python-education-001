import numpy as np

# 1차원 배열
a = np.array([1, 2, 3, 4])
print("1차원 배열:", a)

# 2차원 배열
b = np.array([[1, 2], [3, 4]])
print("2차원 배열:\n", b)

# 배열의 모양
print("배열 shape:", b.shape)

# 기본 배열 생성
zeros = np.zeros((2, 3))
ones = np.ones((3, 3))
arr = np.arange(10)
rand = np.random.rand(2, 2)

print("0으로 채운 배열:\n", zeros)
print("1로 채운 배열:\n", ones)
print("0부터 9까지 배열:", arr)
print("랜덤 배열:\n", rand)

# 통계
print("평균:", arr.mean())
print("합계:", arr.sum())

# 배열 연산
x = np.array([1, 2, 3])
y = np.array([10, 20, 30])
print("덧셈:", x + y)
print("곱셈:", x * y)
print("제곱:", x ** 2)
print("브로드캐스팅:", x + 100)

# 행렬 곱
A = np.array([[1, 2], [3, 4]])
B = np.array([[2, 0], [1, 3]])
C = np.dot(A, B)
print("행렬 곱 결과:\n", C)

# 인덱싱/슬라이싱
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print("두 번째 행:", arr2[1])
print("첫 행, 두 번째 열:", arr2[0, 1])

# 평균 정규화
scores = np.array([80, 90, 100, 70])
normalized = (scores - scores.mean()) / scores.std()
print("정규화된 점수:", normalized)
