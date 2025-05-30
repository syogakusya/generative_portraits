import cv2
import mediapipe as mp
import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

class PortraitExperience:
    def __init__(self, root, video_path=None, on_close_callback=None):
        self.root = root
        self.root.title("ポートレート体験 - コントロール")
        self.on_close_callback = on_close_callback
        
        # 設定
        self.REAL_FACE_WIDTH = 19.0  # 顔の実際の幅（cm）
        self.FOCAL_LENGTH = 600      # カメラの焦点距離（px）
        self.DIST_MAX = 200  # この距離で動画の最初
        self.DIST_MIN = 50.0  # この距離で動画の最後
        
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
        
        # 動画ウィンドウの作成
        self.video_window = tk.Toplevel(self.root)
        self.video_window.title("ポートレート動画")
        self.video_window.configure(bg='black')  # 動画ウィンドウの背景を黒に設定
        self.is_fullscreen = False
        
        # ウィンドウの初期配置
        self.root.geometry("400x600+100+100")  # メインウィンドウ：幅400、高さ600、位置(100,100)
        self.video_window.geometry("820x820+550+100")  # 動画ウィンドウ：幅820、高さ820、位置(550,100)
        
        # GUI要素の作成
        self.create_widgets()
        
        # ウィンドウクローズイベントをバインド
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.video_window.protocol("WM_DELETE_WINDOW", self.quit)
        
        # カメラプレビューの更新
        self.update_camera()
    
    def create_widgets(self):
        # メインウィンドウのUI
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # カメラプレビュー
        self.camera_label = ttk.Label(main_frame)
        self.camera_label.grid(row=0, column=0, padx=5, pady=5)
        
        # 距離表示ラベル
        self.distance_label = ttk.Label(main_frame, text="推定距離: -- cm")
        self.distance_label.grid(row=1, column=0, padx=5, pady=5)
        
        # 動画選択コンボボックス
        self.video_var = tk.StringVar()
        self.video_combo = ttk.Combobox(main_frame, textvariable=self.video_var)
        self.video_combo['values'] = self.get_video_list()
        self.video_combo.grid(row=2, column=0, padx=5, pady=5)
        self.video_combo.bind('<<ComboboxSelected>>', self.on_video_selected)
        
        # フルスクリーン切り替えボタン
        self.fullscreen_btn = ttk.Button(main_frame, text="フルスクリーン切り替え", command=self.toggle_fullscreen)
        self.fullscreen_btn.grid(row=3, column=0, padx=5, pady=5)
        
        # 終了ボタン
        self.quit_btn = ttk.Button(main_frame, text="終了", command=self.quit)
        self.quit_btn.grid(row=4, column=0, padx=5, pady=5)
        
        # 動画ウィンドウのUI
        video_frame = tk.Frame(self.video_window, bg='black', padx=10, pady=10)
        video_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # video_frameの格子設定
        video_frame.grid_rowconfigure(0, weight=1)
        video_frame.grid_columnconfigure(0, weight=1)

        # 動画プレビュー
        self.video_label = tk.Label(video_frame, bg='black')
        self.video_label.grid(row=0, column=0, padx=5, pady=5)
        
        # 動画ウィンドウのキーバインド
        self.video_window.bind('<KeyPress-Escape>', self.exit_fullscreen)
        self.video_window.bind('<KeyPress-F11>', lambda e: self.toggle_fullscreen())
        self.video_window.focus_set()  # フォーカスを設定してキーイベントを受け取れるようにする
    
    def get_video_list(self):
        video_dir = 'Images/out'
        if os.path.exists(video_dir):
            return [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
        return []
    
    def refresh_video_list(self):
        """動画リストを更新する"""
        current_videos = self.get_video_list()
        self.video_combo['values'] = current_videos
        # 現在選択されている動画が削除されていないかチェック
        current_selection = self.video_var.get()
        if current_selection and current_selection not in current_videos:
            # もし現在の選択が無効なら、最初の動画を選択
            if current_videos:
                self.video_var.set(current_videos[0])
                self.on_video_selected(None)
            else:
                self.video_var.set("")
    
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
                # カメラ映像のサイズを調整（コントロールウィンドウに適合）
                frame = cv2.resize(frame, (350, 260))
                image = Image.fromarray(frame)
                photo = ImageTk.PhotoImage(image=image)
                self.camera_label.configure(image=photo)
                self.camera_label.image = photo
                
                # 動画フレームを表示
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                ret, video_frame = self.cap.read()
                if ret:
                    # フルスクリーンかどうかでサイズを調整
                    if self.is_fullscreen:
                        # フルスクリーンの場合は画面サイズに合わせる
                        screen_width = self.video_window.winfo_screenwidth()
                        screen_height = self.video_window.winfo_screenheight()
                        video_frame = cv2.resize(video_frame, (screen_width, screen_height))
                    else:
                        # 通常表示の場合
                        video_frame = cv2.resize(video_frame, (800, 800))
                    
                    video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
                    video_image = Image.fromarray(video_frame)
                    video_photo = ImageTk.PhotoImage(image=video_image)
                    self.video_label.configure(image=video_photo)
                    self.video_label.image = video_photo
        
        # 30ms後に再度更新
        self.root.after(30, self.update_camera)
    
    def quit(self):
        # カメラとビデオキャプチャのリソースを解放
        try:
            if hasattr(self, 'cap_cam') and self.cap_cam.isOpened():
                self.cap_cam.release()
        except:
            pass
            
        try:
            if hasattr(self, 'cap') and self.cap.isOpened():
                self.cap.release()
        except:
            pass
        
        # OpenCVの後処理
        try:
            cv2.destroyAllWindows()
        except:
            pass
        
        # MediaPipeリソースの解放
        try:
            if hasattr(self, 'face_detection'):
                self.face_detection.close()
        except:
            pass
        
        # コールバックを呼び出してPortraitAppに終了を通知
        if self.on_close_callback:
            self.on_close_callback(self)
        
        # 動画ウィンドウも閉じる
        try:
            self.video_window.destroy()
        except:
            pass
        
        try:
            self.root.destroy()
        except:
            pass
    
    def toggle_fullscreen(self):
        """フルスクリーンの切り替え"""
        self.is_fullscreen = not self.is_fullscreen
        self.video_window.attributes('-fullscreen', self.is_fullscreen)
        
        if self.is_fullscreen:
            self.fullscreen_btn.configure(text="フルスクリーン終了")
        else:
            self.fullscreen_btn.configure(text="フルスクリーン切り替え")
    
    def exit_fullscreen(self, event=None):
        """フルスクリーンを終了"""
        if self.is_fullscreen:
            self.is_fullscreen = False
            self.video_window.attributes('-fullscreen', False)
            self.fullscreen_btn.configure(text="フルスクリーン切り替え")

if __name__ == "__main__":
    root = tk.Tk()
    app = PortraitExperience(root)
    root.mainloop()






