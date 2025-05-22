import onnxruntime as rt
import numpy as np

def main():
    # 1) ONNX 세션 로드
    sess = rt.InferenceSession("reduced_pipeline.onnx")

    # 2) 모델 입력 정보 확인
    input_meta = sess.get_inputs()[0]
    input_name = input_meta.name
    expected_shape = input_meta.shape
    print(f"Input name: {input_name}, expected shape: {expected_shape}")

    # 3) 직접 입력할 피처 값 (순서: age, ASBP, sex, exam1 age, smoke, PRSice2)
    #    예시: age=60, ASBP=98, sex=1, exam1 age=47, smoke=0, PRSice2=-0.002597568
    sample = np.array([[66, 160, 1, 50, 0, -0.002517568]], dtype=np.float32)

    # 4) 추론 실행
    outputs = sess.run(None, {input_name: sample})

    # 5) 결과 파싱
    #    outputs[0] -> 예측 라벨(정수), outputs[1] -> 클래스별 확률 배열
    pred_label = outputs[0][0]
    pred_proba = outputs[1][0] if len(outputs) > 1 else None

    print(f"Predicted label: {pred_label}")
    if pred_proba is not None:
        print(f"Predicted probabilities: {pred_proba}")

if __name__ == "__main__":
    main()