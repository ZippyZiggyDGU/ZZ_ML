## 📖 목차

- [설명(Description)](#-설명description)  
- [구성 요소(Components)](#-구성-요소components)  
- [설치(Installation)](#-설치installation)  
- [사용법(Usage)](#-사용법usage)
- [피처 설명(Feature Descriptions)](#-피처-설명feature-descriptions)  
- [출력(Output)](#-출력output)  
 

---

## 📄 설명 (Description)

이 프로젝트는 환자의 건강검진 데이터를 바탕으로 심방세동(AF) 발생 여부를 예측하기 위한 머신러닝 파이프라인을 제공  
- 결측치 대체 및 표준화 전처리
- XGBoost 분류기 학습  
- 전체 파이프라인 ONNX 형식으로 변환  


---

## 🏗️ 구성 요소 (Components)

1. **`preprocessor.joblib`**  
   - `SimpleImputer(mean)` + `StandardScaler`를 적용하는 전처리 파이프라인  
2. **`xgb_model.joblib`**  
   - `XGBClassifier(eval_metric='logloss', random_state=42)` 모델 파일  
3. **`reduced_pipeline.onnx`**  
   - 전처리 + 분류기 전체를 포함하는 ONNX 파이프라인  


---

## ⚙️ 설치 (Installation)

```bash
# 저장소 클론
git clone https://github.com/ZippyZiggyDGU/ZZ_ML.git

# 가상환경 생성 & 활성화
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# 의존성 설치
pip install -r requirements.txt
```

---

## 🚀 사용법 (Usage)

심방세동(AF) 모델 ONNX 파이프라인(`reduced_pipeline.onnx`) 사용법


---

### 1) ONNX 파이프라인 테스트 (`test.py`)

1. **`test.py` 스크립트 위치 확인**  
   프로젝트 최상위에 위치한 `test.py`를 사용합니다. 이 스크립트는 ONNX 런타임을 이용해 모델을 로드하고, 샘플 입력값으로 추론을 수행합니다.

2. **샘플 입력값**  
   - 순서: `age`, `ASBP`, `sex`, `exam1 age`, `smoke`, `PRSice2`  
   - 예시: `66, 160, 1, 50, 0, -0.002517568`

3. **추론 절차**  
   - `InferenceSession`으로 ONNX 모델을 로드합니다.  
   - 런타임에서 요구하는 입력 텐서 이름과 형태(예: `[None, 6]`)를 확인합니다.  
   - NumPy 배열로 샘플 데이터를 준비한 뒤, `sess.run()`에 전달해 예측 결과를 가져옵니다.  
   - `outputs[0]`에는 예측 라벨(`0`: 비발생, `1`: 발생), `outputs[1]`에는 클래스별 확률 배열(`[P(AF=0), P(AF=1)]`)이 담겨 있습니다.

4. **실행 방법**  
   ```bash
   python test.py
   ```

   
---

## 🏡 피처 설명(Feature Descriptions)


1. **age**: 환자의 실제 나이  
2. **ASBP**: 수축기 혈압(최고 혈압) 수치  
   - 최저값 (min): 85.0  
   - 최대값 (max): 191.0  
   - 평균 (mean): 121.261  
   - 중앙값 (median): 120.4  
   - 표준편차 (std): 13.999  
   - 분산 (var): 195.985  

3. **sex**: 성별 (1=남성, 0=여성)  
4. **exam1 age**: 첫 검진 시 나이 기록  
5. **smoke**: 흡연 여부 (1=흡연, 0=비흡연)  
6. **PRSice2**: Polygenic Risk Score (유전위험 점수)  
   - 최저값 (min): -0.002836  
   - 최대값 (max): -0.002481  
   - 평균 (mean): -0.002659  
   - 중앙값 (median): -0.002660  
   - 표준편차 (std): 0.000045 
   - 분산(var) : 0.00000000203


---

## 🖨️ 출력(예시)
```bash
Input name: float_input, expected shape: [None,6]
Predicted label: 1
Predicted probabilities: [0.72 0.28]
```

