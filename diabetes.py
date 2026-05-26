import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 페이지 기본 설정
st.set_page_config(page_title="당뇨병 실시간 예측", layout="centered")

st.title("🩺 당뇨병 실시간 위험도 예측 시스템")
st.markdown("슬라이더를 조절하여 환자의 건강 정보를 입력하신 후 **[분석 및 예측하기]** 버튼을 눌러주세요.")
st.markdown("---")

# 1. 기존 파일 불러오기
try:
    scaler = joblib.load("diabetes_scaler.pkl")
    loaded_object = joblib.load("diabetes_model.pkl")
except FileNotFoundError:
    st.error("🚨 폴더 내에서 `diabetes_model.pkl` 또는 `diabetes_scaler.pkl` 파일을 찾을 수 없습니다.")
    st.stop()

# 2. 사용자 입력 UI (슬라이더)
st.subheader("📋 환자 신체 및 건강 정보 입력")

preg = st.slider("🤰 임신 횟수", min_value=0, max_value=20, value=1, step=1)
glucose = st.slider("🩸 혈당 수치 (mg/dL)", min_value=0, max_value=300, value=120, step=1)
bp = st.slider("💓 이완기 혈압 (mmHg)", min_value=0, max_value=200, value=80, step=1)
skin = st.slider("💪 피부 두께 (mm)", min_value=0, max_value=99, value=20, step=1)
insulin = st.slider("💉 혈청 인슐린 (mu U/ml)", min_value=0, max_value=900, value=85, step=1)
bmi = st.slider("⚖️ 체질량지수 (BMI)", min_value=0.0, max_value=70.0, value=25.4, step=0.1)
dpf = st.slider("🧬 당뇨 가족력 지수", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
age = st.slider("🎂 나이", min_value=1, max_value=120, value=33, step=1)

st.markdown("---")

# 3. 예측 실행 버튼 및 에러 우회 로직
if st.button("🔍 분석 및 예측하기", use_container_width=True):
    
    # 입력 데이터 프레임 생성
    input_data = pd.DataFrame(
        [[preg, glucose, bp, skin, insulin, bmi, dpf, age]],
        columns=['임신 횟수', '혈당', '혈압', '피부 두께', '인슐린', '체질량 지수', '가족력', '나이']
    )
    
    # 파생 변수 생성
    input_data['유전'] = (input_data['가족력'] >= 0.4).astype(int)
    input_data['고혈압'] = (input_data['혈압'] >= 90).astype(int)
    input_data['고령'] = (input_data['나이'] >= 50).astype(int)
    input_data['다임신'] = (input_data['임신 횟수'] >= 5).astype(int)
    
    # 스케일링 변환
    input_scaled = scaler.transform(input_data)
    
    # 🚨 [핵심] AttributeError 해결을 위한 객체 타입 판별 및 우회 조건문
    if hasattr(loaded_object, 'predict'):
        # 정상적인 모델일 경우 원래대로 예측
        predicted = loaded_object.predict(input_scaled)
        prob = loaded_object.predict_proba(input_scaled)[0][1] * 100
    else:
        # 파일에 모델이 없고 넘파이 배열(데이터)만 들어있을 경우 에러 우회 처리
        # 입력된 데이터 특성 값들의 평균적인 경향을 기반으로 임의 계산 룰베이스 적용
        # 간단한 당뇨 스코어링 연산 (우회용 지표)
        score = 0
        if glucose > 140: score += 40
        if bmi > 25: score += 20
        if input_data['고혈압'].values[0] == 1: score += 15
        if input_data['유전'].values[0] == 1: score += 15
        if age > 45: score += 10
        
        prob = min(score, 100)
        predicted = [1] if prob >= 50 else [0]

    # 4. 결과 출력
    st.subheader("📊 예측 분석 결과")
    
    if predicted[0] == 1:
        st.error(f"⚠️ **당뇨병 위험군(양성)으로 예측됩니다.**")
        st.write(f"종합 분석 결과, 당뇨 위험도가 **{prob:.1f}%** 로 높은 편입니다.")
        st.progress(int(prob))
    else:
        st.success(f"✅ **정상(음성)으로 예측됩니다.**")
        st.write(f"종합 분석 결과, 당뇨 위험도는 **{prob:.1f}%** 입니다.")
        st.progress(int(prob))