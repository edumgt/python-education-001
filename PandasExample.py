import pandas as pd
import numpy as np

# 1. 예제 데이터 생성
data = {
    '이름': ['철수', '영희', '민수', '지민', '수진', '도윤'],
    '학년': [1, 1, 2, 2, 2, 1],
    '수학': [90, 85, np.nan, 75, 82, 88],
    '영어': [np.nan, 92, 80, 78, np.nan, 85]
}

df = pd.DataFrame(data)
print("📌 원본 데이터:")
print(df)

# 2. 결측치 처리
print("\n📌 결측치 개수:")
print(df.isnull().sum())

# 평균값으로 결측치 채우기
df['수학'] = df['수학'].fillna(df['수학'].mean())
df['영어'] = df['영어'].fillna(df['영어'].mean())

print("\n📌 결측치 처리 후:")
print(df)

# 3. 학년별 평균 점수 (groupby)
grouped = df.groupby('학년')[['수학', '영어']].mean()
print("\n📊 학년별 평균 점수:")
print(grouped)

# 4. 결측치 포함 행 제거 예시 (선택적)
df_dropped = df.dropna()
print("\n📌 결측치 제거한 데이터:")
print(df_dropped)
