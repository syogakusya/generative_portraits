import cv2
import mediapipe as mp
import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

class PortraitExperience:
    def __init__(self, root, video_path=None):
        self.root = root
        self.root.title("ポートレート体験")
        
        # 設定
        self.REAL_FACE_WIDTH = 16.0  # 顔の実際の幅（cm）
        self.FOCAL_LENGTH = 600      # カメラの焦点距離（px）
        self.DIST_MAX = 100  # この距離で動画の最初
        self.DIST_MIN = 60.0  # この距離で動画の最後
        
        # MediaPipeのセットアップ
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, 
            min_detection_confidence=0.3
        )
        
        # カメラの設定
        self.cap_cam = cv2.VideoCapture(1)
        
        # 動画の設定
        self.video_path = video_path if video_path else os.path.join('Images/out', 'output.mp4')
        self.cap = cv2.VideoCapture(self.video_path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        
        # GUI要素の作成
        self.create_widgets()
        
        # カメラプレビューの更新
        self.update_camera()
    
    def create_widgets(self):
        # メインフレーム
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # カメラプレビュー
        self.camera_label = ttk.Label(main_frame)
        self.camera_label.grid(row=0, column=0, padx=5, pady=5)
        
        # 動画プレビュー
        self.video_label = ttk.Label(main_frame)
        self.video_label.grid(row=0, column=1, padx=5, pady=5)
        
        # 距離表示ラベル
        self.distance_label = ttk.Label(main_frame, text="推定距離: -- cm")
        self.distance_label.grid(row=1, column=0, columnspan=2, padx=5, pady=5)
        
        # 動画選択コンボボックス
        self.video_var = tk.StringVar()
        self.video_combo = ttk.Combobox(main_frame, textvariable=self.video_var)
        self.video_combo['values'] = self.get_video_list()
        self.video_combo.grid(row=2, column=0, columnspan=2, padx=5, pady=5)
        self.video_combo.bind('<<ComboboxSelected>>', self.on_video_selected)
        
        # 終了ボタン
        self.quit_btn = ttk.Button(main_frame, text="終了", command=self.quit)
        self.quit_btn.grid(row=3, column=0, columnspan=2, padx=5, pady=5)
    
    def get_video_list(self):
        video_dir = 'Images/out'
        if os.path.exists(video_dir):
            return [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
        return []
    
    def on_video_selected(self, event):
        selected = self.video_var.get()
        if selected:
            self.video_path = os.path.join('Images/out', selected)
            self.cap.release()
            self.cap = cv2.VideoCapture(self.video_path)
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.current_frame = 0
    
    def update_camera(self):
        if self.cap_cam.isOpened():
            ret, frame = self.cap_cam.read()
            if ret:
                # 顔検出用にRGB変換
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_detection.process(frame_rgb)
                
                if results.detections:
                    for detection in results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        ih, iw, _ = frame.shape
                        face_width_pixel = bboxC.width * iw
                        
                        if face_width_pixel > 0:
                            distance_cm = (self.REAL_FACE_WIDTH * self.FOCAL_LENGTH) / face_width_pixel
                            self.distance_label.configure(text=f"推定距離: {distance_cm:.2f} cm")
                            
                            # 動画の再生位置を距離に応じて決定
                            clamped_distance = max(min(distance_cm, self.DIST_MAX), self.DIST_MIN)
                            normalized = (self.DIST_MAX - clamped_distance) / (self.DIST_MAX - self.DIST_MIN)
                            self.current_frame = int(normalized * (self.total_frames - 1))
                
                # カメラ映像を表示
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
                photo = ImageTk.PhotoImage(image=image)
                self.camera_label.configure(image=photo)
                self.camera_label.image = photo
                
                # 動画フレームを表示
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                ret, video_frame = self.cap.read()
                if ret:
                    video_frame = cv2.resize(video_frame, (1200, 1200))
                    video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
                    video_image = Image.fromarray(video_frame)
                    video_photo = ImageTk.PhotoImage(image=video_image)
                    self.video_label.configure(image=video_photo)
                    self.video_label.image = video_photo
        
        # 30ms後に再度更新
        self.root.after(30, self.update_camera)
    
    def quit(self):
        self.cap_cam.release()
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = PortraitExperience(root)
    root.mainloop()






