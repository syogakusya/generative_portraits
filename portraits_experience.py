import cv2
import mediapipe as mp
import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import math
from tkinter import colorchooser
import serial
import threading
import time

## todo
# MINMAX距離をスライダーで変更できるように
# mediapipeのモデルを更新

class PortraitExperience:
    def __init__(self, root, video_path=None, on_close_callback=None):
        self.root = root
        self.root.title("ポートレート体験 - コントロール")
        self.on_close_callback = on_close_callback
        
        # 設定
        self.REAL_FACE_WIDTH = 16.0  # 顔の実際の幅（cm）
        self.FOCAL_LENGTH = 800.0    # カメラの焦点距離（px）
        self.DIST_MAX = 100.0  # この距離で動画の最初
        self.DIST_MIN = 60.0  # この距離で動画の最後
        
        # デバッグモード
        self.debug_mode = False
        self.manual_distance = 100.0  # デバッグモードでの手動距離
        
        # Arduino関連の設定
        self.use_arduino = False
        self.arduino_serial = None
        self.arduino_distance = 0.0
        self.arduino_thread = None
        self.arduino_running = False
        
        # 顔認識状態
        self.face_detected = False
        self.last_face_time = 0.0
        
        # MediaPipeのセットアップ
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, 
            min_detection_confidence=0.3
        )
        
        # カメラの設定（固定カメラID使用）
        self.portrait_camera_id = 1  # ポートレート体験用カメラID（デフォルト1）
        self.cap_cam = cv2.VideoCapture(self.portrait_camera_id)
        
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
        self.root.geometry("400x750+200+200")  # メインウィンドウ：幅400、高さ750、位置(200,200)
        self.video_window.geometry("820x820+550+200")  # 動画ウィンドウ：幅820、高さ820、位置(550,200)
        
        # 背景色の初期値を設定（create_widgetsの前に定義）
        self.bg_color = 'black'
        
        # GUI要素の作成
        self.create_widgets()
        
        # ウィンドウクローズイベントをバインド
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.video_window.protocol("WM_DELETE_WINDOW", self.quit)
        
        # カメラプレビューの更新
        self.update_camera()
    
    def find_available_camera(self):
        """利用可能なカメラを検出（GUI appで使用中のカメラを避ける）"""
        # カメラ0は通常GUI appで使用されているので、1から順番に試す
        for camera_id in range(1, 5):  # カメラ1-4を試す
            cap = cv2.VideoCapture(camera_id)
            if cap.isOpened():
                # テスト撮影して正常に動作するか確認
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"カメラ {camera_id} を使用します")
                    return cap
                cap.release()
        
        # カメラ1-4が利用できない場合、カメラ0を試す（共有使用）
        print("警告: 専用カメラが見つかりません。カメラ0を共有使用します")
        return cv2.VideoCapture(0)
    
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
        
        # 顔認識状態表示ラベル
        self.face_status_label = ttk.Label(main_frame, text="顔認識: 未検出")
        self.face_status_label.grid(row=2, column=0, padx=5, pady=5)
        
        # カメラ設定フレーム
        camera_frame = ttk.LabelFrame(main_frame, text="カメラ設定", padding="5")
        camera_frame.grid(row=3, column=0, padx=5, pady=10, sticky=(tk.W, tk.E))
        
        # カメラ選択モード
        self.camera_mode_var = tk.StringVar(value="固定")
        ttk.Label(camera_frame, text="カメラ選択:").grid(row=0, column=0, padx=5)
        self.camera_mode_combo = ttk.Combobox(camera_frame, textvariable=self.camera_mode_var, 
                                            values=["固定", "自動検出"], state="readonly", width=10)
        self.camera_mode_combo.grid(row=0, column=1, padx=5)
        self.camera_mode_combo.bind('<<ComboboxSelected>>', self.on_camera_mode_changed)
        
        # カメラID選択（固定モード用）
        ttk.Label(camera_frame, text="カメラID:").grid(row=1, column=0, padx=5)
        self.portrait_camera_var = tk.StringVar(value="1")
        self.portrait_camera_combo = ttk.Combobox(camera_frame, textvariable=self.portrait_camera_var, 
                                                values=["0", "1", "2", "3", "4"], state="readonly", width=5)
        self.portrait_camera_combo.grid(row=1, column=1, padx=5)
        self.portrait_camera_combo.bind('<<ComboboxSelected>>', self.on_portrait_camera_changed)
        
        # カメラテストボタン
        self.test_portrait_camera_btn = ttk.Button(camera_frame, text="カメラテスト", command=self.test_portrait_camera)
        self.test_portrait_camera_btn.grid(row=1, column=2, padx=5)
        
        # 自動検出ボタン
        self.auto_detect_btn = ttk.Button(camera_frame, text="自動検出実行", command=self.auto_detect_camera)
        self.auto_detect_btn.grid(row=0, column=2, padx=5)
        self.auto_detect_btn.grid_remove()  # 初期状態では非表示
        
        # カメラ状態表示ラベル
        camera_status = "接続OK" if self.cap_cam.isOpened() else "接続エラー"
        self.camera_status_label = ttk.Label(camera_frame, text=f"カメラ{self.portrait_camera_id}: {camera_status}")
        self.camera_status_label.grid(row=2, column=0, columnspan=3, padx=5, pady=2)
        
        # デバッグモード切り替えチェックボックス
        self.debug_var = tk.BooleanVar()
        self.debug_checkbox = ttk.Checkbutton(main_frame, text="デバッグモード", 
                                             variable=self.debug_var, 
                                             command=self.toggle_debug_mode)
        self.debug_checkbox.grid(row=4, column=0, padx=5, pady=5)
        
        # 距離調整スライダー（デバッグモード用）
        self.distance_var = tk.DoubleVar(value=100.0)
        self.distance_slider = ttk.Scale(main_frame, from_=self.DIST_MIN, to=self.DIST_MAX,
                                       variable=self.distance_var, orient=tk.HORIZONTAL,
                                       command=self.on_distance_changed)
        self.distance_slider.grid(row=5, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))
        self.distance_slider.grid_remove()  # 初期状態では非表示
        
        # スライダー値表示ラベル
        self.slider_label = ttk.Label(main_frame, text="手動距離: 100.0 cm")
        self.slider_label.grid(row=6, column=0, padx=5, pady=5)
        self.slider_label.grid_remove()  # 初期状態では非表示
        
        # Arduino設定フレーム
        arduino_frame = ttk.LabelFrame(main_frame, text="Arduino設定", padding="5")
        arduino_frame.grid(row=8, column=0, padx=5, pady=10, sticky=(tk.W, tk.E))
        
        # Arduino使用チェックボックス
        self.arduino_var = tk.BooleanVar()
        self.arduino_checkbox = ttk.Checkbutton(arduino_frame, text="Arduino超音波センサーを使用", 
                                               variable=self.arduino_var, 
                                               command=self.toggle_arduino)
        self.arduino_checkbox.grid(row=0, column=0, columnspan=2, padx=5, pady=5)
        
        # COMポート選択
        ttk.Label(arduino_frame, text="COMポート:").grid(row=1, column=0, padx=5, pady=5)
        self.com_var = tk.StringVar(value="COM3")
        self.com_entry = ttk.Entry(arduino_frame, textvariable=self.com_var, width=10)
        self.com_entry.grid(row=1, column=1, padx=5, pady=5)
        
        # Arduino距離表示
        self.arduino_distance_label = ttk.Label(arduino_frame, text="Arduino距離: -- cm")
        self.arduino_distance_label.grid(row=2, column=0, columnspan=2, padx=5, pady=5)
        
        # 動画選択コンボボックス
        self.video_var = tk.StringVar()
        self.video_combo = ttk.Combobox(main_frame, textvariable=self.video_var)
        self.video_combo['values'] = self.get_video_list()
        self.video_combo.grid(row=9, column=0, padx=5, pady=5)
        self.video_combo.bind('<<ComboboxSelected>>', self.on_video_selected)
        
        # フルスクリーン切り替えボタン
        self.fullscreen_btn = ttk.Button(main_frame, text="フルスクリーン切り替え", command=self.toggle_fullscreen)
        self.fullscreen_btn.grid(row=10, column=0, padx=5, pady=5)
        
        # ディスプレイ選択コンボボックス
        display_frame = ttk.Frame(main_frame)
        display_frame.grid(row=11, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        ttk.Label(display_frame, text="ディスプレイ:").grid(row=0, column=0, padx=(0, 5))
        self.display_var = tk.StringVar()
        self.display_combo = ttk.Combobox(display_frame, textvariable=self.display_var, state="readonly", width=25)
        display_names = [d["name"] for d in self.displays]
        self.display_combo['values'] = display_names
        self.display_combo.current(0)  # デフォルトでメインディスプレイを選択
        self.display_combo.grid(row=0, column=1)
        self.display_combo.bind('<<ComboboxSelected>>', self.on_display_selected)
        
        # 背景色選択ボタン
        self.bgcolor_btn = ttk.Button(main_frame, text="背景色を選択", command=self.choose_bg_color)
        self.bgcolor_btn.grid(row=12, column=0, padx=5, pady=5)
        
        # キャリブレーションボタン
        self.calibrate_btn = ttk.Button(main_frame, text="カメラキャリブレーション", command=self.open_calibration_window)
        self.calibrate_btn.grid(row=13, column=0, padx=5, pady=5)
        
        # 終了ボタン
        self.quit_btn = ttk.Button(main_frame, text="終了", command=self.quit)
        self.quit_btn.grid(row=14, column=0, padx=5, pady=5)
        
        # 動画ウィンドウのUI
        self.video_frame = tk.Frame(self.video_window, bg=self.bg_color, padx=10, pady=10)
        self.video_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # video_frameの格子設定
        self.video_frame.grid_rowconfigure(0, weight=1)
        self.video_frame.grid_columnconfigure(0, weight=1)

        # 動画プレビュー
        self.video_label = tk.Label(self.video_frame, bg=self.bg_color)
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
        # カメラが正常に動作しているかチェック
        if not self.cap_cam.isOpened():
            self.camera_status_label.configure(text=f"カメラ{self.portrait_camera_id}: 接続エラー")
            self.face_status_label.configure(text="顔認識: カメラ未接続")
            # 30ms後に再度更新
            self.root.after(30, self.update_camera)
            return
            
        if self.cap_cam.isOpened():
            ret, frame = self.cap_cam.read()
            if ret:
                # 顔検出用にRGB変換
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_detection.process(frame_rgb)
                
                # 顔認識状態をリセット
                self.face_detected = False
                
                # デバッグモードでない場合のみ顔検出を実行
                if not self.debug_mode:
                    if results.detections:
                        self.face_detected = True
                        self.last_face_time = time.time()
                        
                        for detection in results.detections:
                            bboxC = detection.location_data.relative_bounding_box
                            ih, iw, _ = frame.shape
                            face_width_pixel = bboxC.width * iw
                            
                            if face_width_pixel > 0:
                                # Arduino使用時は顔認識中のみArduino距離を使用、それ以外は顔サイズから計算
                                if self.use_arduino and self.arduino_distance > 0:
                                    distance_cm = self.arduino_distance
                                    self.distance_label.configure(text=f"Arduino距離: {distance_cm:.2f} cm")
                                else:
                                    distance_cm = (self.REAL_FACE_WIDTH * self.FOCAL_LENGTH) / face_width_pixel
                                    self.distance_label.configure(text=f"推定距離: {distance_cm:.2f} cm")
                                
                                # 動画の再生位置を距離に応じて決定
                                clamped_distance = max(min(distance_cm, self.DIST_MAX), self.DIST_MIN)
                                normalized = (self.DIST_MAX - clamped_distance) / (self.DIST_MAX - self.DIST_MIN)
                                self.current_frame = int(normalized * (self.total_frames - 1))
                    
                    # 顔認識状態の更新
                    if self.face_detected:
                        self.face_status_label.configure(text="顔認識: 検出中")
                    else:
                        self.face_status_label.configure(text="顔認識: 未検出")
                else:
                    # デバッグモードの場合は手動距離を使用
                    distance_cm = self.manual_distance
                    self.distance_label.configure(text=f"手動距離: {distance_cm:.2f} cm")
                    self.face_detected = True  # デバッグモードでは常に顔ありとして扱う
                    
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
                
                # 動画表示の処理
                self.update_video_display()
        
        # 30ms後に再度更新
        self.root.after(30, self.update_camera)
    
    def update_video_display(self):
        """動画表示の更新"""
        # 顔が検出されている場合のみ動画を表示
        if self.face_detected:
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
                    video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
                    video_image = Image.fromarray(video_frame)
                    
                    screen_width = self.video_window.winfo_screenwidth()
                    screen_height = self.video_window.winfo_screenheight()
                    background = Image.new('RGB', (screen_width, screen_height), self.bg_color)
                    x = (screen_width - video_image.width) // 2
                    y = (screen_height - video_image.height) // 2
                    background.paste(video_image, (x, y))
                    video_photo = ImageTk.PhotoImage(image=background)
                else:
                    # 通常表示の場合
                    video_frame = cv2.resize(video_frame, (800, 800))
                    video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
                    video_image = Image.fromarray(video_frame)
                    video_photo = ImageTk.PhotoImage(image=video_image)
                
                self.video_label.configure(image=video_photo)
                self.video_label.image = video_photo
        else:
            # 顔が検出されていない場合は背景色のみ表示
            if self.is_fullscreen:
                screen_width = self.video_window.winfo_screenwidth()
                screen_height = self.video_window.winfo_screenheight()
                background = Image.new('RGB', (screen_width, screen_height), self.bg_color)
            else:
                background = Image.new('RGB', (800, 800), self.bg_color)
            
            video_photo = ImageTk.PhotoImage(image=background)
            self.video_label.configure(image=video_photo)
            self.video_label.image = video_photo
    
    def quit(self):
        # Arduino接続を終了
        self.disconnect_arduino()
        
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

    def on_portrait_camera_changed(self, event):
        """ポートレート用カメラが変更されたときの処理"""
        new_camera_id = int(self.portrait_camera_var.get())
        
        # 現在のカメラを解放
        if self.cap_cam.isOpened():
            self.cap_cam.release()
        
        # 新しいカメラを開く
        self.portrait_camera_id = new_camera_id
        self.cap_cam = cv2.VideoCapture(self.portrait_camera_id)
        
        if self.cap_cam.isOpened():
            self.camera_status_label.configure(text=f"カメラ{self.portrait_camera_id}: 接続成功")
            print(f"ポートレート用カメラをカメラ{self.portrait_camera_id}に変更しました")
        else:
            self.camera_status_label.configure(text=f"カメラ{self.portrait_camera_id}: 接続失敗")
            print(f"カメラ{self.portrait_camera_id}への接続に失敗しました")
    
    def test_portrait_camera(self):
        """ポートレート用カメラの接続テスト"""
        camera_id = int(self.portrait_camera_var.get())
        test_cap = cv2.VideoCapture(camera_id)
        
        if test_cap.isOpened():
            ret, frame = test_cap.read()
            if ret and frame is not None:
                self.camera_status_label.configure(text=f"カメラ{camera_id}: テスト成功")
                print(f"カメラ{camera_id}のテストに成功しました")
            else:
                self.camera_status_label.configure(text=f"カメラ{camera_id}: 読み取り失敗")
                print(f"カメラ{camera_id}からのデータ読み取りに失敗しました")
            test_cap.release()
        else:
            self.camera_status_label.configure(text=f"カメラ{camera_id}: 接続失敗")
            print(f"カメラ{camera_id}への接続に失敗しました")
    
    def on_camera_mode_changed(self, event):
        """カメラ選択モードが変更されたときの処理"""
        mode = self.camera_mode_var.get()
        if mode == "自動検出":
            # 自動検出ボタンを表示、カメラID選択とテストボタンを非表示
            self.auto_detect_btn.grid()
            self.portrait_camera_combo.configure(state="disabled")
            self.test_portrait_camera_btn.grid_remove()
        else:
            # 固定モード：カメラID選択とテストボタンを表示、自動検出ボタンを非表示
            self.auto_detect_btn.grid_remove()
            self.portrait_camera_combo.configure(state="readonly")
            self.test_portrait_camera_btn.grid()
    
    def auto_detect_camera(self):
        """利用可能なカメラを自動検出して設定"""
        # 現在のカメラを解放
        if self.cap_cam.isOpened():
            self.cap_cam.release()
        
        # 自動検出実行
        detected_camera = self.find_available_camera()
        if detected_camera.isOpened():
            self.cap_cam = detected_camera
            self.camera_status_label.configure(text="自動検出: 成功")
            print("カメラの自動検出に成功しました")
        else:
            self.camera_status_label.configure(text="自動検出: 失敗")
            print("利用可能なカメラが見つかりませんでした")

    def choose_bg_color(self):
        color_code = colorchooser.askcolor(title="背景色を選択")
        if color_code and color_code[1]:
            self.bg_color = color_code[1]
            self.video_window.configure(bg=self.bg_color)
            self.video_frame.configure(bg=self.bg_color)
            self.video_label.configure(bg=self.bg_color)

    def toggle_arduino(self):
        """Arduino使用の切り替え"""
        self.use_arduino = self.arduino_var.get()
        if self.use_arduino:
            self.connect_arduino()
        else:
            self.disconnect_arduino()
    
    def connect_arduino(self):
        """Arduinoとの接続を開始"""
        try:
            com_port = self.com_var.get()
            self.arduino_serial = serial.Serial(com_port, 9600, timeout=1)
            time.sleep(2)  # Arduino初期化待機
            
            # Arduino通信スレッドを開始
            self.arduino_running = True
            self.arduino_thread = threading.Thread(target=self.arduino_communication_thread)
            self.arduino_thread.daemon = True
            self.arduino_thread.start()
            
            print(f"Arduino接続成功: {com_port}")
        except Exception as e:
            print(f"Arduino接続エラー: {e}")
            self.arduino_var.set(False)
            self.use_arduino = False
    
    def disconnect_arduino(self):
        """Arduinoとの接続を終了"""
        self.arduino_running = False
        if self.arduino_thread:
            self.arduino_thread.join(timeout=1)
        if self.arduino_serial:
            self.arduino_serial.close()
            self.arduino_serial = None
        print("Arduino接続を終了")
    
    def arduino_communication_thread(self):
        """Arduino通信スレッド"""
        while self.arduino_running and self.arduino_serial:
            try:
                if self.arduino_serial.in_waiting > 0:
                    line = self.arduino_serial.readline().decode('utf-8').strip()
                    if line:
                        # 距離データを取得（例：「Distance: 25.4」形式を想定）
                        if line.startswith('Distance:'):
                            distance_str = line.split(':')[1].strip()
                            self.arduino_distance = float(distance_str)
                            self.arduino_distance_label.configure(text=f"Arduino距離: {self.arduino_distance:.1f} cm")
                time.sleep(0.1)
            except Exception as e:
                print(f"Arduino通信エラー: {e}")
                break

if __name__ == "__main__":
    root = tk.Tk()
    app = PortraitExperience(root)
    root.mainloop()






