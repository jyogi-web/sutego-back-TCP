from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import cv2
import mediapipe as mp
import json
import struct
import socket
import threading

app = FastAPI()

# MediaPipeの初期化
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    smooth_segmentation=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# カメラの初期化
cap = cv2.VideoCapture(0)  # 必要に応じてデバイスIDを変更

if not cap.isOpened():
    print("カメラが開けませんでした")
    exit()

# IPアドレスの取得
hostname = socket.gethostname()
local_ip = socket.gethostbyname(hostname)
print("local_ipを取得：" + str(local_ip))

# サーバーのホストとポートを設定
HOST = local_ip
PORT_IMAGE = 50007  # キャプチャ画像用
PORT_POSE = 50008  # 座標データ用

# TCP/IPソケットを作成し、サーバーに接続
def setup_tcp_sockets():
    print("画像データ用TCP/IPソケットを作成しています")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT_IMAGE))
    print("画像データ用ソケット作成完了")

    print("座標データ用TCP/IPソケットを作成しています")
    s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s2.bind((HOST, PORT_POSE))
    print("座標データ用ソケット作成完了")

    print("クライアントの接続を待ち受け中...")
    s.listen(1)
    conn, addr = s.accept()
    print("画像データ用クライアント接続成功")
    print('Connected by', addr)

    s2.listen(1)
    conn2, addr2 = s2.accept()
    print("座標データ用クライアント接続成功")
    print('connected by', addr2)

    return conn, conn2

conn, conn2 = setup_tcp_sockets()

@app.get("/")
def read_root():
    return {"Hello": "World"}

def process_and_send_data():
    try:
        print("キャプチャした画像を転送中...")
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("フレームがキャプチャできませんでした")
                break

            # BGR画像をRGBに変換
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = pose.process(frame_rgb)
            frame_rgb.flags.writeable = True
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                pose_landmarks = {}
                for id, landmark in enumerate(results.pose_landmarks.landmark):
                    pose_landmarks[f'{id}'] = [
                        round(landmark.x, 3),
                        round(landmark.y, 3),
                        round(landmark.z, 3)
                    ]
                mp_drawing.draw_landmarks(
                    frame_bgr,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

                # 画像データのエンコード
                _, buffer = cv2.imencode('.jpg', frame_bgr)
                byte_data = buffer.tobytes()

                # 画像データとJSONデータの送信
                conn.sendall(struct.pack('>I', len(byte_data)) + byte_data)
                conn.sendall(byte_data)
                json_data = json.dumps(pose_landmarks).encode('utf-8')
                conn2.sendall(json_data)

            # ESCキーが押されたらループを終了
            if cv2.waitKey(1) & 0xFF == 27:
                break
    except Exception as e:
        print(f"送信中にエラーが発生しました: {e}")
    finally:
        cap.release()
        conn.close()
        conn2.close()
        print("カメラとソケットを閉じました")

# データ処理と送信を別スレッドで実行
threading.Thread(target=process_and_send_data).start()

@app.websocket("/ws/data")
async def websocket_data_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        print("キャプチャした画像をWebSocketで転送中...")
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("フレームがキャプチャできませんでした")
                break

            # BGR画像をRGBに変換
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = pose.process(frame_rgb)
            frame_rgb.flags.writeable = True
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                pose_landmarks = {}
                for id, landmark in enumerate(results.pose_landmarks.landmark):
                    pose_landmarks[f'{id}'] = [
                        round(landmark.x, 3),
                        round(landmark.y, 3),
                        round(landmark.z, 3)
                    ]
                mp_drawing.draw_landmarks(
                    frame_bgr,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

                # 画像データのエンコード
                _, buffer = cv2.imencode('.jpg', frame_bgr)
                byte_data = buffer.tobytes()

                # 画像データとJSONデータの送信
                await websocket.send_bytes(struct.pack('>I', len(byte_data)) + byte_data)
                await websocket.send_text(json.dumps(pose_landmarks))

            # ESCキーが押されたらループを終了
            if cv2.waitKey(1) & 0xFF == 27:
                break
    except WebSocketDisconnect:
        print("WebSocketが切断されました")
    except Exception as e:
        print(f"送信中にエラーが発生しました: {e}")
    finally:
        cap.release()
        print("カメラを閉じました")

@app.on_event("shutdown")
def shutdown_event():
    cap.release()
    print("サーバーをシャットダウンしました")
