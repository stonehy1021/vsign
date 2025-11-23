import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import mediapipe as mp
import av
import numpy as np
import time
import queue

# ---------------- 1. 기본 설정 ----------------
st.set_page_config(page_title="Smart Selfie (Face + V-sign)", layout="centered")
st.title("📸 Smart Selfie")
st.markdown("얼굴과 **브이(V) 포즈**를 인식하면 3초 뒤 자동으로 찍어줍니다! ✌️")

# 세션 상태 초기화
if "snapshot" not in st.session_state:
    st.session_state.snapshot = None

# Mediapipe 초기화
mp_face = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ---------------- 2. 헬퍼 함수: V포즈 인식 ----------------
def is_victory(lms, w, h):
    """손가락 좌표를 분석해 V 포즈인지 확인"""
    def c(i):
        lm = lms.landmark[i]
        return int(lm.x * w), int(lm.y * h)

    # 손가락 끝(tip)과 마디(knuckle) 좌표
    i_tip, m_tip = c(8), c(12)  # 검지, 중지 끝
    r_tip, p_tip = c(16), c(20) # 약지, 새끼 끝
    i_kn, m_kn = c(5), c(9)     # 검지, 중지 마디
    r_kn, p_kn = c(13), c(17)   # 약지, 새끼 마디

    # 검지와 중지는 펴져 있고(끝이 마디보다 위), 나머지는 접혀 있어야 함 (좌표계상 위가 y값이 작음)
    # 하지만 손 방향에 따라 다를 수 있으므로 단순하게 상대적 위치 비교
    # 여기서는 손이 위를 향할 때 기준으로 작성됨 (일반적인 V)
    
    # 펴짐 조건: 팁이 관절보다 위에 있음 (y값이 작음)
    index_open = i_tip[1] < i_kn[1]
    middle_open = m_tip[1] < m_kn[1]
    
    # 접힘 조건: 팁이 관절보다 아래에 있음 (y값이 큼)
    ring_folded = r_tip[1] > r_kn[1]
    pinky_folded = p_tip[1] > p_kn[1]

    return index_open and middle_open and ring_folded and pinky_folded

# ---------------- 3. 영상 처리 클래스 ----------------
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        # 모델 로드
        self.face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)
        self.hand_detector = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6)
        
        self.result_queue = queue.Queue() # 메인 스레드로 이미지를 보낼 통로
        self.capture_triggered = False
        self.enter_time = None
        self.flash_frame = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # 거울 모드
        h, w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 1. 얼굴 감지
        face_res = self.face_detector.process(rgb)
        face_detected = face_res.detections is not None

        # 2. 손 감지 및 V 포즈 확인
        hand_res = self.hand_detector.process(rgb)
        victory_detected = False

        if hand_res.multi_hand_landmarks:
            for handLms in hand_res.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
                if is_victory(handLms, w, h):
                    victory_detected = True
                    break
        
        if face_detected:
            for d in face_res.detections:
                mp_draw.draw_detection(img, d)

        # 3. 로직 판정 (얼굴 + V포즈)
        status_msg = "Show Face & V-sign"
        color = (0, 0, 255) # 빨강

        # 플래시 효과
        if self.flash_frame > 0:
            self.flash_frame -= 1
            white = np.full((h, w, 3), 255, dtype=np.uint8)
            img = cv2.addWeighted(img, 0.5, white, 0.5, 0)

        if face_detected and victory_detected:
            color = (0, 255, 0) # 초록
            status_msg = "HOLD ON!"
            
            # 카운트다운 로직
            if self.enter_time is None:
                self.enter_time = time.time()
            
            elapsed = time.time() - self.enter_time
            countdown = 1.5 - elapsed # 1.5초 대기
            
            if countdown > 0:
                cv2.putText(img, f"{countdown:.1f}", (w//2-50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 4)
            else:
                # 촬영 시점
                if not self.capture_triggered:
                    self.result_queue.put(img) # 큐에 이미지 넣기
                    self.capture_triggered = True
                    self.flash_frame = 5
        else:
            self.enter_time = None
            self.capture_triggered = False

        # 상태 메시지 출력
        cv2.putText(img, status_msg, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---------------- 4. UI 구성 ----------------

# 4-1. 결과 화면 (촬영 후)
if st.session_state.snapshot is not None:
    st.success("📸 촬영 성공!")
    st.image(st.session_state.snapshot, channels="BGR", caption="내 V라인 샷", use_container_width=True)
    
    # 다운로드 버튼
    img_rgb = cv2.cvtColor(st.session_state.snapshot, cv2.COLOR_BGR2RGB) # 저장용 변환
    is_success, buffer = cv2.imencode(".jpg", st.session_state.snapshot) # OpenCV 기본이 BGR이라 그대로 인코딩
    
    if is_success:
        st.download_button(
            label="📥 사진 다운로드",
            data=buffer.tobytes(),
            file_name=f"V_Selfie_{int(time.time())}.jpg",
            mime="image/jpeg",
            type="primary"
        )
    
    st.warning("🔄 다시 찍으려면 페이지를 새로고침 해주세요!")

# 4-2. 촬영 화면
else:
    ctx = webrtc_streamer(
        key="v-sign-camera",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": {"facingMode": "user"}, "audio": False},
    )

    # 큐 확인 루프 (자동 촬영 감지)
    if ctx.state.playing:
        while True:
            if ctx.video_processor:
                try:
                    # 프로세서에서 보낸 이미지가 있는지 확인
                    result_img = ctx.video_processor.result_queue.get(timeout=0.1)
                    if result_img is not None:
                        st.session_state.snapshot = result_img
                        st.rerun() # 화면 갱신
                except queue.Empty:
                    pass
            time.sleep(0.1)
