import cv2
import mediapipe as mp
import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import math

class PortraitExperience:
    def __init__(self, root, video_path=None, on_close_callback=None):
        self.root = root
        self.root.title("ポートレート体験 - コントロール")
        self.on_close_callback = on_close_callback
        
        # 設定
        self.REAL_FACE_WIDTH = 16.0  # 顔の実際の幅（cm）
        self.FOCAL_LENGTH = 800.0   # カメラの焦点距離（px）
        self.DIST_MAX = 170.0  # この距離で動画の最初
        self.DIST_MIN = 120.0  # この距離で動画の最後
        
        # デバッグモード
        self.debug_mode = False
        self.manual_distance = 100.0  # デバッグモードでの手動距離
        
        # 顔検出結果を保存する変数
        self.latest_detection_result = None
        
        # MediaPipe Tasks APIのセットアップ
        BaseOptions = mp.tasks.BaseOptions
        FaceDetector = mp.tasks.vision.FaceDetector
        FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        
        # コールバック関数を定義
        def result_callback(result: mp.tasks.vision.FaceDetectorResult, output_image: mp.Image, timestamp_ms: int):
            self.latest_detection_result = result
        
        # 顔検出の設定（デフォルトモデルを使用）
        options = FaceDetectorOptions(
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=result_callback,
            min_detection_confidence=0.3
        )
        
        self.detector = FaceDetector.create_from_options(options)
        
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
        
        # ディスプレイ情報を取得
        self.detect_displays()
        self.selected_display = 0  # デフォルトはメインディスプレイ
        
        # ウィンドウの初期配置
        self.root.geometry("400x700+200+200")  # メインウィンドウ：幅400、高さ700、位置(200,200)
        self.video_window.geometry("820x820+550+200")  # 動画ウィンドウ：幅820、高さ820、位置(550,200)
        
        # GUI要素の作成
        self.create_widgets()
        
        # ウィンドウクローズイベントをバインド
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.video_window.protocol("WM_DELETE_WINDOW", self.quit)
        
        # カメラプレビューの更新
        self.update_camera()
    
    def detect_displays(self):
        """利用可能なディスプレイを検出"""
        # メインディスプレイの情報を取得
        main_width = self.root.winfo_screenwidth()
        main_height = self.root.winfo_screenheight()
        
        # 仮想デスクトップの全体サイズを取得
        virtual_width = self.root.winfo_vrootwidth()
        virtual_height = self.root.winfo_vrootheight()
        
        # 簡易的な2ディスプレイ検出
        # 横に並んでいる場合
        if virtual_width > main_width:
            self.displays = [
                {"name": "メインディスプレイ", "x": 0, "y": 0, "width": main_width, "height": main_height},
                {"name": "セカンドディスプレイ", "x": main_width, "y": 0, "width": virtual_width - main_width, "height": main_height}
            ]
        # 縦に並んでいる場合
        elif virtual_height > main_height:
            self.displays = [
                {"name": "メインディスプレイ", "x": 0, "y": 0, "width": main_width, "height": main_height},
                {"name": "セカンドディスプレイ", "x": 0, "y": main_height, "width": main_width, "height": virtual_height - main_height}
            ]
        else:
            self.displays = [
                {"name": "メインディスプレイ", "x": 0, "y": 0, "width": main_width, "height": main_height}
            ]
        
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
        
        # デバッグモード切り替えチェックボックス
        self.debug_var = tk.BooleanVar()
        self.debug_checkbox = ttk.Checkbutton(main_frame, text="デバッグモード", 
                                             variable=self.debug_var, 
                                             command=self.toggle_debug_mode)
        self.debug_checkbox.grid(row=2, column=0, padx=5, pady=5)
        
        # 距離調整スライダー（デバッグモード用）
        self.distance_var = tk.DoubleVar(value=100.0)
        self.distance_slider = ttk.Scale(main_frame, from_=self.DIST_MIN, to=self.DIST_MAX,
                                       variable=self.distance_var, orient=tk.HORIZONTAL,
                                       command=self.on_distance_changed)
        self.distance_slider.grid(row=3, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))
        self.distance_slider.grid_remove()  # 初期状態では非表示
        
        # スライダー値表示ラベル
        self.slider_label = ttk.Label(main_frame, text="手動距離: 100.0 cm")
        self.slider_label.grid(row=4, column=0, padx=5, pady=5)
        self.slider_label.grid_remove()  # 初期状態では非表示
        
        # 動画選択コンボボックス
        self.video_var = tk.StringVar()
        self.video_combo = ttk.Combobox(main_frame, textvariable=self.video_var)
        self.video_combo['values'] = self.get_video_list()
        self.video_combo.grid(row=5, column=0, padx=5, pady=5)
        self.video_combo.bind('<<ComboboxSelected>>', self.on_video_selected)
        
        # フルスクリーン切り替えボタン
        self.fullscreen_btn = ttk.Button(main_frame, text="フルスクリーン切り替え", command=self.toggle_fullscreen)
        self.fullscreen_btn.grid(row=6, column=0, padx=5, pady=5)
        
        # ディスプレイ選択コンボボックス
        display_frame = ttk.Frame(main_frame)
        display_frame.grid(row=7, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        ttk.Label(display_frame, text="ディスプレイ:").grid(row=0, column=0, padx=(0, 5))
        self.display_var = tk.StringVar()
        self.display_combo = ttk.Combobox(display_frame, textvariable=self.display_var, state="readonly", width=25)
        display_names = [d["name"] for d in self.displays]
        self.display_combo['values'] = display_names
        self.display_combo.current(0)  # デフォルトでメインディスプレイを選択
        self.display_combo.grid(row=0, column=1)
        self.display_combo.bind('<<ComboboxSelected>>', self.on_display_selected)
        
        # 終了ボタン
        self.quit_btn = ttk.Button(main_frame, text="終了", command=self.quit)
        self.quit_btn.grid(row=8, column=0, padx=5, pady=5)
        
        # キャリブレーションボタン
        self.calibrate_btn = ttk.Button(main_frame, text="カメラキャリブレーション", command=self.open_calibration_window)
        self.calibrate_btn.grid(row=9, column=0, padx=5, pady=5)
        
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
                # デバッグモードでない場合のみ顔検出を実行
                if not self.debug_mode:
                    # MediaPipe Tasks API用にフレームを変換
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                    
                    # 顔検出を非同期で実行（タイムスタンプを指定）
                    timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
                    self.detector.detect_async(mp_image, timestamp_ms)
                    
                    # 最新の検出結果を使用
                    if self.latest_detection_result and self.latest_detection_result.detections:
                        # 最初の顔の境界ボックスを取得
                        detection = self.latest_detection_result.detections[0]
                        bbox = detection.bounding_box
                        
                        # 顔の幅をピクセル単位で計算
                        ih, iw, _ = frame.shape
                        face_width_pixel = bbox.width * iw
                        
                        if face_width_pixel > 0:
                            distance_cm = (self.REAL_FACE_WIDTH * self.FOCAL_LENGTH) / face_width_pixel
                            self.distance_label.configure(text=f"推定距離: {distance_cm:.2f} cm")
                            
                            # 動画の再生位置を距離に応じて決定
                            clamped_distance = max(min(distance_cm, self.DIST_MAX), self.DIST_MIN)
                            normalized = (self.DIST_MAX - clamped_distance) / (self.DIST_MAX - self.DIST_MIN)
                            self.current_frame = int(normalized * (self.total_frames - 1))
                    else:
                        self.distance_label.configure(text="推定距離: 顔が検出されていません")
                else:
                    # デバッグモードの場合は手動距離を使用
                    distance_cm = self.manual_distance
                    self.distance_label.configure(text=f"手動距離: {distance_cm:.2f} cm")
                    
                    # 動画の再生位置を手動距離に応じて決定
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
                        # フルスクリーンの場合は画面サイズに合わせつつアスペクト比を保持
                        screen_width = self.video_window.winfo_screenwidth()
                        screen_height = self.video_window.winfo_screenheight()
                        
                        # 元の動画のアスペクト比を計算
                        h, w = video_frame.shape[:2]
                        aspect_ratio = w / h
                        
                        # スクリーンのアスペクト比を計算
                        screen_aspect_ratio = screen_width / screen_height
                        
                        if screen_aspect_ratio > aspect_ratio:
                            # スクリーンの方が横長の場合、高さに合わせる
                            new_height = screen_height
                            new_width = int(screen_height * aspect_ratio)
                        else:
                            # スクリーンの方が縦長の場合、幅に合わせる
                            new_width = screen_width
                            new_height = int(screen_width / aspect_ratio)
                        
                        video_frame = cv2.resize(video_frame, (new_width, new_height))
                    else:
                        # 通常表示の場合
                        video_frame = cv2.resize(video_frame, (800, 800))
                    
                    video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
                    video_image = Image.fromarray(video_frame)
                    
                    if self.is_fullscreen:
                        # フルスクリーン時は黒背景の画像を作成して、その中央に動画を配置
                        screen_width = self.video_window.winfo_screenwidth()
                        screen_height = self.video_window.winfo_screenheight()
                        background = Image.new('RGB', (screen_width, screen_height), 'black')
                        
                        # 動画を中央に配置するための座標を計算
                        x = (screen_width - video_image.width) // 2
                        y = (screen_height - video_image.height) // 2
                        
                        # 動画を黒背景の中央に配置
                        background.paste(video_image, (x, y))
                        video_photo = ImageTk.PhotoImage(image=background)
                    else:
                        # 通常表示の場合は直接表示
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
            if hasattr(self, 'detector'):
                self.detector.close()
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
    
    def on_display_selected(self, event):
        """ディスプレイが選択されたときの処理"""
        self.selected_display = self.display_combo.current()
        # フルスクリーンでない場合でも、選択したディスプレイに移動
        if not self.is_fullscreen:
            display = self.displays[self.selected_display]
            # ウィンドウを選択したディスプレイの中央に配置
            window_width = 820
            window_height = 820
            x = display["x"] + (display["width"] - window_width) // 2
            y = display["y"] + (display["height"] - window_height) // 2
            self.video_window.geometry(f"{window_width}x{window_height}+{x}+{y}")
    
    def toggle_fullscreen(self):
        """フルスクリーンの切り替え"""
        self.is_fullscreen = not self.is_fullscreen
        
        if self.is_fullscreen:
            # 選択されたディスプレイの情報を取得
            display = self.displays[self.selected_display]
            
            # ウィンドウを選択したディスプレイに移動
            self.video_window.geometry(f"+{display['x']}+{display['y']}")
            
            # 少し待ってからフルスクリーンにする（ウィンドウの移動を確実にするため）
            self.video_window.after(100, lambda: self.video_window.attributes('-fullscreen', True))
            self.fullscreen_btn.configure(text="フルスクリーン終了")
        else:
            self.video_window.attributes('-fullscreen', False)
            # 元のサイズに戻す
            display = self.displays[self.selected_display]
            window_width = 820
            window_height = 820
            x = display["x"] + (display["width"] - window_width) // 2
            y = display["y"] + (display["height"] - window_height) // 2
            self.video_window.geometry(f"{window_width}x{window_height}+{x}+{y}")
            self.fullscreen_btn.configure(text="フルスクリーン切り替え")
    
    def exit_fullscreen(self, event=None):
        """フルスクリーンを終了"""
        if self.is_fullscreen:
            self.is_fullscreen = False
            self.video_window.attributes('-fullscreen', False)
            # 元のサイズに戻す
            display = self.displays[self.selected_display]
            window_width = 820
            window_height = 820
            x = display["x"] + (display["width"] - window_width) // 2
            y = display["y"] + (display["height"] - window_height) // 2
            self.video_window.geometry(f"{window_width}x{window_height}+{x}+{y}")
            self.fullscreen_btn.configure(text="フルスクリーン切り替え")
    
    def toggle_debug_mode(self):
        """デバッグモードの切り替え"""
        self.debug_mode = self.debug_var.get()
        if self.debug_mode:
            # デバッグモード有効：スライダーを表示
            self.distance_slider.grid()
            self.slider_label.grid()
        else:
            # デバッグモード無効：スライダーを非表示
            self.distance_slider.grid_remove()
            self.slider_label.grid_remove()
    
    def on_distance_changed(self, value):
        """スライダーの値が変更されたときの処理"""
        self.manual_distance = float(value)
        self.slider_label.configure(text=f"手動距離: {self.manual_distance:.1f} cm")

    def open_calibration_window(self):
        # キャリブレーションウィンドウを開く処理を実装
        pass

if __name__ == "__main__":
    root = tk.Tk()
    app = PortraitExperience(root)
    root.mainloop()






