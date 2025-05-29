# 라이브러리 불러오기
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#1,2 데이터 불러오기
df = pd.read_csv("한국_기업문화_HR_데이터셋_샘플.csv")

#3 결측치 확인 및 제거
print("결측치 수:\n", df.isnull().sum())
df.dropna(inplace=True)

# 범주형 변수 인코딩
label_encoders = {}
for col in df.select_dtypes(include="object").columns:
    if col != "이직여부":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# 이직여부를 이진값으로 변환
df["이직여부"] = df["이직여부"].map({"Yes": 1, "No": 0})

#4 선택한 피처 목록 및 선택 이유
selected_features = [
    "월급여",            # 보상이 낮으면 이직 가능성 높음
    "업무만족도",        # 직무 만족도는 이직의 핵심 요인
    "이전회사경험수",    # 잦은 이직 경험은 향후 이직 가능성 시사
    "Age",              # 연령대별로 이직 경향이 다를 수 있음
    "야근여부",          # 과도한 야근은 이직을 유도
    "근무환경만족도",    # 사내 환경이 만족스럽지 않으면 이직 고려 가능
    "워라밸",            # 균형 부족은 이직 사유
    "현회사근속년수",    # 근속기간이 짧을수록 이직 가능성 높음
    "직무"               # 특정 직무군이 이직률이 높을 수 있음
]

#5 입력값(X), 목표값(y)
X = df[selected_features]
y = df["이직여부"]

# 학습/검증 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

#6 성능 검증
y_pred = model.predict(X_test)
print("정확도:", accuracy_score(y_test, y_pred))
print("혼동 행렬:\n", confusion_matrix(y_test, y_pred))
# 정확도 해석: 약 84.5%의 정확도로 이직 여부를 예측할 수 있음. 
# 이는 실제 인사 데이터에서도 의미 있는 정확도로 간주됨.
# 혼동행렬 분석
# 이직을 예측하는 모델로서는 개선 여지가 있음. 이직자 대부분을 비이직으로 예측하고 있어서 실무에서 이직 리스크 탐지에는 부적합할 수 있다.
# 정밀도는 높은 편이지만 재현율이 낮아 이직자를 놓칠 위험이 큼.

#7 예측 결과 분석
df["예측값"] = model.predict(X)
df["이직확률"] = model.predict_proba(X)[:, 1]

print("\n이직 여부가 1(Yes)로 예측된 직원 수:", (df["예측값"] == 1).sum())

print("\n이직 가능성 높은 상위 5명:")
print(df[selected_features + ["이직확률"]].sort_values(by="이직확률", ascending=False).head(5))

#8 신규 입사자 예측용 데이터
new_data = pd.DataFrame([
    {
        "Age": 29, "출장빈도": "Travel_Rarely", "부서": "Research & Development",
        "집까지거리": 5, "학력수준": 3, "전공분야": "Life Sciences",
        "근무환경만족도": 2, "성별": "Male", "시간당급여": 70,
        "업무몰입도": 3, "직급": 1, "직무": "Laboratory Technician",
        "업무만족도": 2, "결혼상태": "Single", "월급여": 2800,
        "이전회사경험수": 1, "야근여부": "Yes", "연봉인상률": 12,
        "성과등급": 3, "대인관계만족도": 2, "스톡옵션등급": 0,
        "총경력": 4, "연간교육횟수": 2, "워라밸": 2,
        "현회사근속년수": 1, "현직무근속년수": 1, "최근승진후경과년수": 0,
        "현상사근속년수": 1
    },
    {
        "Age": 42, "출장빈도": "Non-Travel", "부서": "Human Resources",
        "집까지거리": 10, "학력수준": 4, "전공분야": "Human Resources",
        "근무환경만족도": 3, "성별": "Female", "시간당급여": 85,
        "업무몰입도": 3, "직급": 3, "직무": "Human Resources",
        "업무만족도": 4, "결혼상태": "Married", "월급여": 5200,
        "이전회사경험수": 2, "야근여부": "No", "연봉인상률": 14,
        "성과등급": 3, "대인관계만족도": 3, "스톡옵션등급": 1,
        "총경력": 18, "연간교육횟수": 3, "워라밸": 3,
        "현회사근속년수": 7, "현직무근속년수": 4, "최근승진후경과년수": 1,
        "현상사근속년수": 3
    },
    {
        "Age": 35, "출장빈도": "Travel_Frequently", "부서": "Sales",
        "집까지거리": 2, "학력수준": 2, "전공분야": "Marketing",
        "근무환경만족도": 1, "성별": "Male", "시간당급여": 65,
        "업무몰입도": 2, "직급": 2, "직무": "Sales Executive",
        "업무만족도": 1, "결혼상태": "Single", "월급여": 3300,
        "이전회사경험수": 3, "야근여부": "Yes", "연봉인상률": 11,
        "성과등급": 3, "대인관계만족도": 2, "스톡옵션등급": 0,
        "총경력": 10, "연간교육횟수": 2, "워라밸": 2,
        "현회사근속년수": 2, "현직무근속년수": 1, "최근승진후경과년수": 1,
        "현상사근속년수": 1
    }
])

# 신규 데이터에도 동일한 인코딩 적용
for col, le in label_encoders.items():
    if col in new_data.columns:
        new_data[col] = le.transform(new_data[col])

new_input = new_data[selected_features]
preds = model.predict(new_input)
print("\n신규 입사자 이직 예측 결과:", preds.tolist())

#9 피처 중요도 분석
importances = model.feature_importances_
features = X_train.columns
importance = pd.Series(importances, index=features).sort_values(ascending=False)

print("\n상위 3개 중요 피처:")
print(importance.head(3))

# 실무적 해석 
print("\n[상위 피처 실무적 해석]")
for feat in importance.head(3).index:
    if feat == "업무만족도":
        print(f"- {feat}: 구성원이 자신의 업무에 만족하지 못하면 스트레스가 누적되고, 성장 가능성에 대한 회의감이 생겨 이직을 고려하게 됩니다. 실제로 다양한 조직행동 연구에서 직무만족도는 이직의 주요 예측 변수로 나타납니다.")
    elif feat == "월급여":
        print(f"- {feat}: 보상 수준은 직무에 대한 보람이나 소속감보다 더 강력하게 이직 결정을 좌우하는 경우가 많습니다. 특히 유사한 직무에서 더 나은 급여를 제시하는 경우 이직 가능성이 높아집니다.")
    elif feat == "야근여부":
        print(f"- {feat}: 지속적인 야근은 근로자의 삶의 질을 저하시키고, 건강 악화 및 가족/사회관계 단절을 초래할 수 있습니다. 워라밸(Work-Life Balance)에 대한 관심이 높아진 최근에는 이직을 유도하는 핵심 요인 중 하나입니다.")
    elif feat == "직무":
        print(f"- {feat}: 직무별로 요구되는 스트레스 수준이나 이직률 자체가 구조적으로 차이가 납니다. 예컨대 판매나 상담 직군은 높은 감정노동으로 인해 이직률이 높게 나타나는 경향이 있습니다.")
    elif feat == "현회사근속년수":
        print(f"- {feat}: 근속년수가 짧은 경우, 회사와의 유대감이 약하고 업무에 대한 몰입이 낮아 이직 가능성이 높을 수 있습니다. 특히 1~2년 차는 이직 위험이 높은 시기입니다.")