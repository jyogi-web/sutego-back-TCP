import FastAPI
import socket
import threading
import cv2 # type: ignore
import struct
import mediapipe as mp
import json

app = FastAPI()

@app.get("/hello")
async def hello():
    return {"message": "hello world!"}

def handle_tcp_image_client(conn, pose, mp_drawing, mp_drawing_styles, cap):
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            results = pose.process(frame)
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            pose_landmarks = {}
            if results.pose_landmarks:
                for id, landmark in enumerate(results.pose_landmarks.landmark):
                    pose_landmarks[f'{id}'] = [
                        round(landmark.x, 3),
                        round(landmark.y, 3),
                        round(landmark.z, 3)
                    ]
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            _, buffer = cv2.imencode('.jpg', frame)
            byte_data = buffer.tobytes()
            conn.sendall(struct.pack('>I', len(byte_data)))
            conn.sendall(byte_data)

            json_data = json.dumps(pose_landmarks).encode('utf-8')
            conn2.sendall(json_data)

            if cv2.waitKey(1) & 0xFF == 27:
                break
    except Exception as e:
        print(f"Error during transmission: {e}")
    finally:
        conn.close()
        cap.release()

def start_tcp_server(host, port, pose, mp_drawing, mp_drawing_styles):
    cap = cv2.VideoCapture(0)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen(1)
    print(f"TCP Server Listening on port {port}...")

    while True:
        conn, addr = server.accept()
        print(f"Accepted connection from {addr}")
        client_handler = threading.Thread(target=handle_tcp_image_client, args=(conn, pose, mp_drawing, mp_drawing_styles, cap))
        client_handler.start()

@app.on_event("startup")
def startup_event():
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print("Local IP:", local_ip)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    PORT_IMAGE = 50007

    threading.Thread(target=start_tcp_server, args=(local_ip, PORT_IMAGE, pose, mp_drawing, mp_drawing_styles)).start()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
