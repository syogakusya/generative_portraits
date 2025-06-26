import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import os
import time
import torch
import threading
import queue
import gc  # ガベージコレクション用
from collections import OrderedDict
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer

class PortraitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ポートレート生成アプリ")
        
        # カメラ設定
        self.gui_camera_id = 0  # GUI app用カメラID
        self.cap = cv2.VideoCapture(self.gui_camera_id)
        self.is_capturing = False
        self.camera_paused = False  # カメラ一時停止フラグ
        
        # 画像生成キュー
        self.image_queue = queue.Queue(maxsize=10)  # 最大10件まで
        self.processing = False
        self.processing_lock = threading.Lock()  # 処理の重複を防ぐ
        
        # 生成済み動画のリスト
        self.generated_videos = []
        
        # PortraitExperienceのインスタンス参照
        self.experience_instances = []
        
        # モデルを初期化時にはロードしない（遅延ロード）
        self.model = None
        self.data_loader = None
        self.dataset = None
        self.visualizer = None
        self.opt = None
        
        # 既存動画のリストを読み込む
        self.load_existing_videos()
        
        # GUI要素の作成
        self.create_widgets()
        
        # カメラプレビューの更新
        self.update_camera()
        
        # 画像処理スレッドの開始
        self.start_processing_thread()
    
    def load_existing_videos(self):
        """既存の動画ファイルを読み込む"""
        video_dir = 'Images/out'
        if os.path.exists(video_dir):
            for file in os.listdir(video_dir):
                if file.endswith('.mp4'):
                    self.generated_videos.append(os.path.join(video_dir, file))
    
    def load_existing_images(self):
        """既存の画像まとめファイル、論文用画像、画像フレームディレクトリを読み込む"""
        image_dir = 'Images/out'
        if os.path.exists(image_dir):
            for item in os.listdir(image_dir):
                item_path = os.path.join(image_dir, item)
                
                # 画像まとめファイル
                if item.startswith('combined_') and item.endswith('.png'):
                    self.generated_files.append(("画像まとめ", item_path))
                    self.video_listbox.insert(tk.END, f"[画像まとめ] {item}")
                
                # 論文用画像ファイル
                elif item.startswith('paper_') and item.endswith('.png'):
                    if item.startswith('paper_gen_'):
                        self.generated_files.append(("論文用画像（生成のみ）", item_path))
                        self.video_listbox.insert(tk.END, f"[論文用画像（生成のみ）] {item}")
                    else:
                        self.generated_files.append(("論文用画像", item_path))
                        self.video_listbox.insert(tk.END, f"[論文用画像] {item}")
                
                # 画像フレームディレクトリ
                elif item.startswith('frames_') and os.path.isdir(item_path):
                    self.generated_files.append(("画像フレーム", item_path))
                    self.video_listbox.insert(tk.END, f"[画像フレーム] {item}")
    
    def open_selected_file(self):
        """選択されたファイルを開く"""
        selected_indices = self.video_listbox.curselection()
        if selected_indices:
            file_type, file_path = self.generated_files[selected_indices[0]]
            try:
                if os.name == 'nt':  # Windows
                    os.startfile(file_path)
                elif os.name == 'posix':  # macOS and Linux
                    import subprocess
                    subprocess.call(['open', file_path] if os.uname().sysname == 'Darwin' else ['xdg-open', file_path])
            except Exception as e:
                self.status_label.configure(text=f"ファイルを開けませんでした: {e}")
    
    def open_output_folder(self):
        """出力フォルダを開く"""
        output_dir = 'Images/out'
        try:
            if os.name == 'nt':  # Windows
                os.startfile(output_dir)
            elif os.name == 'posix':  # macOS and Linux
                import subprocess
                subprocess.call(['open', output_dir] if os.uname().sysname == 'Darwin' else ['xdg-open', output_dir])
        except Exception as e:
            self.status_label.configure(text=f"フォルダを開けませんでした: {e}")
        
    def setup_model(self):
        """遅延ロード：実際に必要になったときのみモデルをロード"""
        if self.model is not None:
            return  # 既にロード済み
            
        # モデルの初期化
        self.opt = TestOptions().parse(save=False)
        self.opt.display_id = 0
        self.opt.gpu_ids = [0]
        self.opt.nThreads = 1
        self.opt.batchSize = 1
        self.opt.serial_batches = True
        self.opt.no_flip = True
        self.opt.in_the_wild = True
        self.opt.traverse = True
        self.opt.interp_step = 0.05
        self.opt.no_moving_avg = True
        self.opt.fineSize = 256
        
        # 選択された性別に応じてモデル名と画像リストファイルを設定
        selected_gender = self.gender_var.get()
        if selected_gender == "女性":
            self.opt.name = 'females_model'
            self.opt.image_path_file = 'females_image_list.txt'
        else:  # 男性
            self.opt.name = 'males_model'
            self.opt.image_path_file = 'males_image_list.txt'
            
        print(f"モデル名を設定しました: {self.opt.name}")
        print(f"画像リストファイル: {self.opt.image_path_file}")
        
        self.data_loader = CreateDataLoader(self.opt)
        self.dataset = self.data_loader.load_data()
        self.visualizer = Visualizer(self.opt)
        
        # 現在選択されている背景色を適用
        selected = self.background_var.get()
        color_map = {"黒": 0, "グレー": 128, "白": 255}
        background_color = color_map.get(selected, 0)
        self.dataset.dataset.set_background_color(background_color)
        
        self.model = create_model(self.opt)
        self.model.eval()
        
        print(f"使用中のモデル: {self.opt.name}")
        print(f"checkpoints/{self.opt.name} からモデルをロードしました")
        
        # GPUメモリ最適化
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("モデルのロードが完了しました")
    
    def cleanup_model(self):
        """モデルとリソースを明示的に解放"""
        try:
            if self.model is not None:
                # モデルをCPUに移動してからメモリ解放
                if hasattr(self.model, 'netG'):
                    if hasattr(self.model.netG, 'cpu'):
                        self.model.netG.cpu()
                if hasattr(self.model, 'netD'):
                    if hasattr(self.model.netD, 'cpu'):
                        self.model.netD.cpu()
                        
                del self.model
                self.model = None
            
            if self.data_loader is not None:
                del self.data_loader
                self.data_loader = None
            
            if self.dataset is not None:
                del self.dataset
                self.dataset = None
                
            if self.visualizer is not None:
                del self.visualizer
                self.visualizer = None
            
            # optも初期化（モデル切り替え時のために）
            if self.opt is not None:
                del self.opt
                self.opt = None
            
            # GPUメモリの徹底的なクリア
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            
            gc.collect()
            print("モデルとリソースを解放しました")
            
        except Exception as e:
            print(f"モデル解放時にエラーが発生: {e}")
    
    def create_widgets(self):
        # メインフレーム
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 左側カラム（カメラとファイルリスト）
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, padx=5, pady=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # カメラプレビュー
        self.camera_label = ttk.Label(left_frame)
        self.camera_label.grid(row=0, column=0, padx=2, pady=2)
        
        # ボタンフレーム
        button_frame = ttk.Frame(left_frame)
        button_frame.grid(row=1, column=0, padx=2, pady=2)
        
        # 撮影ボタン
        self.capture_btn = ttk.Button(button_frame, text="撮影", command=self.capture_image)
        self.capture_btn.grid(row=0, column=0, padx=2)
        
        # 画像選択ボタン
        self.select_image_btn = ttk.Button(button_frame, text="画像を選択", command=self.select_image)
        self.select_image_btn.grid(row=0, column=1, padx=2)
        
        # 体験開始ボタン
        self.experience_btn = ttk.Button(button_frame, text="体験開始", command=self.start_experience)
        self.experience_btn.grid(row=0, column=2, padx=2)
        
        # 生成済みファイルリスト
        list_frame = ttk.Frame(left_frame)
        list_frame.grid(row=2, column=0, padx=2, pady=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # リストボックスのラベル
        self.list_label = ttk.Label(list_frame, text="生成済みファイル：")
        self.list_label.grid(row=0, column=0, sticky=tk.W)
        
        # ボタンフレーム
        list_button_frame = ttk.Frame(list_frame)
        list_button_frame.grid(row=0, column=1, padx=2, sticky=tk.E)
        
        # ファイルを開くボタン
        self.open_file_btn = ttk.Button(list_button_frame, text="ファイルを開く", command=self.open_selected_file)
        self.open_file_btn.grid(row=0, column=0, padx=1)
        
        # フォルダを開くボタン
        self.open_folder_btn = ttk.Button(list_button_frame, text="フォルダを開く", command=self.open_output_folder)
        self.open_folder_btn.grid(row=0, column=1, padx=1)
        
        # スクロールバー付きリストボックス
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self.video_listbox = tk.Listbox(list_frame, height=4, yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.video_listbox.yview)
        
        self.video_listbox.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=1, column=2, sticky=(tk.N, tk.S))
        
        # 右側カラム（設定とステータス）
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, padx=5, pady=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 設定グループ1
        settings1_frame = ttk.LabelFrame(right_frame, text="基本設定", padding="5")
        settings1_frame.grid(row=0, column=0, padx=2, pady=2, sticky=(tk.W, tk.E))
        
        # 性別選択
        ttk.Label(settings1_frame, text="性別:").grid(row=0, column=0, padx=2, sticky=tk.W)
        self.gender_var = tk.StringVar(value="女性")
        self.gender_combo = ttk.Combobox(settings1_frame, textvariable=self.gender_var, 
                                        values=["女性", "男性"], state="readonly", width=8)
        self.gender_combo.grid(row=0, column=1, padx=2, sticky=tk.W)
        self.gender_combo.bind('<<ComboboxSelected>>', self.on_gender_changed)
        
        # 背景色選択
        ttk.Label(settings1_frame, text="背景色:").grid(row=1, column=0, padx=2, sticky=tk.W)
        self.background_var = tk.StringVar(value="黒")
        self.background_combo = ttk.Combobox(settings1_frame, textvariable=self.background_var, 
                                           values=["黒", "グレー", "白"], state="readonly", width=8)
        self.background_combo.grid(row=1, column=1, padx=2, sticky=tk.W)
        self.background_combo.bind('<<ComboboxSelected>>', self.on_background_changed)
        
        # 設定グループ2
        settings2_frame = ttk.LabelFrame(right_frame, text="出力設定", padding="5")
        settings2_frame.grid(row=1, column=0, padx=2, pady=2, sticky=(tk.W, tk.E))
        
        # 出力形式選択
        ttk.Label(settings2_frame, text="出力形式:").grid(row=0, column=0, padx=2, sticky=tk.W)
        self.output_format_var = tk.StringVar(value="動画のみ")
        self.output_format_combo = ttk.Combobox(settings2_frame, textvariable=self.output_format_var, 
                                              values=["動画のみ", "画像フレームのみ", "画像まとめのみ", "論文用画像", "論文用画像（生成のみ）", "動画+画像フレーム", "動画+画像まとめ", "動画+論文用画像", "すべて"], 
                                              state="readonly", width=20)
        self.output_format_combo.grid(row=0, column=1, padx=2, sticky=tk.W)
        
        # カメラ設定グループ
        camera_frame = ttk.LabelFrame(right_frame, text="カメラ設定", padding="5")
        camera_frame.grid(row=2, column=0, padx=2, pady=2, sticky=(tk.W, tk.E))
        
        ttk.Label(camera_frame, text="カメラID:").grid(row=0, column=0, padx=2, sticky=tk.W)
        self.gui_camera_var = tk.StringVar(value="0")
        self.gui_camera_combo = ttk.Combobox(camera_frame, textvariable=self.gui_camera_var, 
                                           values=["0", "1", "2", "3", "4"], state="readonly", width=5)
        self.gui_camera_combo.grid(row=0, column=1, padx=2, sticky=tk.W)
        self.gui_camera_combo.bind('<<ComboboxSelected>>', self.on_gui_camera_changed)
        
        # カメラテストボタン
        self.test_camera_btn = ttk.Button(camera_frame, text="テスト", command=self.test_gui_camera)
        self.test_camera_btn.grid(row=0, column=2, padx=2)
        
        # カメラ状態表示
        self.gui_camera_status = ttk.Label(camera_frame, text="カメラ0: 接続中")
        self.gui_camera_status.grid(row=1, column=0, columnspan=3, padx=2, pady=1)
        
        # ステータスグループ
        status_frame = ttk.LabelFrame(right_frame, text="ステータス", padding="5")
        status_frame.grid(row=3, column=0, padx=2, pady=2, sticky=(tk.W, tk.E))
        
        # 進捗バー
        self.progress = ttk.Progressbar(status_frame, length=180, mode='determinate')
        self.progress.grid(row=0, column=0, padx=2, pady=2, sticky=(tk.W, tk.E))
        
        # ステータスラベル
        self.status_label = ttk.Label(status_frame, text="準備完了")
        self.status_label.grid(row=1, column=0, padx=2, pady=2, sticky=(tk.W, tk.E))
        
        # メモリクリアボタン（デバッグ用）
        self.cleanup_btn = ttk.Button(status_frame, text="メモリクリア", command=self.manual_cleanup)
        self.cleanup_btn.grid(row=2, column=0, padx=2, pady=2)
        
        # 生成済みファイルリスト（動画と画像）
        self.generated_files = []
        
        # 既存の動画を表示
        for video_path in self.generated_videos:
            self.generated_files.append(("動画", video_path))
            self.video_listbox.insert(tk.END, f"[動画] {os.path.basename(video_path)}")
        
        # 既存の画像まとめファイルも読み込む
        self.load_existing_images()
        
        # グリッドの重みを設定
        main_frame.grid_columnconfigure(0, weight=2)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        left_frame.grid_rowconfigure(2, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)
    
    def pause_camera(self):
        """カメラを一時停止（Portrait Experience起動時）"""
        self.camera_paused = True
        if self.cap.isOpened():
            self.cap.release()
        print("GUI アプリのカメラを一時停止しました")
    
    def resume_camera(self):
        """カメラを再開（Portrait Experience終了時）"""
        self.camera_paused = False
        try:
            self.cap = cv2.VideoCapture(self.gui_camera_id)
            if self.cap.isOpened():
                print(f"GUI アプリのカメラ{self.gui_camera_id}を再開しました")
                self.gui_camera_status.configure(text=f"カメラ{self.gui_camera_id}: 接続中")
            else:
                print(f"警告: カメラ{self.gui_camera_id}の再開に失敗しました")
                self.gui_camera_status.configure(text=f"カメラ{self.gui_camera_id}: 接続失敗")
        except Exception as e:
            print(f"カメラ再開エラー: {e}")
            self.gui_camera_status.configure(text=f"カメラ{self.gui_camera_id}: エラー")
    
    def on_gui_camera_changed(self, event):
        """GUI用カメラが変更されたときの処理"""
        new_camera_id = int(self.gui_camera_var.get())
        
        # 現在のカメラを解放
        if self.cap.isOpened():
            self.cap.release()
        
        # 新しいカメラを開く
        self.gui_camera_id = new_camera_id
        self.cap = cv2.VideoCapture(self.gui_camera_id)
        
        if self.cap.isOpened():
            self.gui_camera_status.configure(text=f"カメラ{self.gui_camera_id}: 接続成功")
            print(f"GUI用カメラをカメラ{self.gui_camera_id}に変更しました")
        else:
            self.gui_camera_status.configure(text=f"カメラ{self.gui_camera_id}: 接続失敗")
            print(f"カメラ{self.gui_camera_id}への接続に失敗しました")
    
    def test_gui_camera(self):
        """GUI用カメラの接続テスト"""
        camera_id = int(self.gui_camera_var.get())
        test_cap = cv2.VideoCapture(camera_id)
        
        if test_cap.isOpened():
            ret, frame = test_cap.read()
            if ret and frame is not None:
                self.gui_camera_status.configure(text=f"カメラ{camera_id}: テスト成功")
                print(f"カメラ{camera_id}のテストに成功しました")
            else:
                self.gui_camera_status.configure(text=f"カメラ{camera_id}: 読み取り失敗")
                print(f"カメラ{camera_id}からのデータ読み取りに失敗しました")
            test_cap.release()
        else:
            self.gui_camera_status.configure(text=f"カメラ{camera_id}: 接続失敗")
            print(f"カメラ{camera_id}への接続に失敗しました")
    
    def manual_cleanup(self):
        """手動でメモリクリアを実行"""
        self.cleanup_model()
        self.status_label.configure(text="メモリクリアを実行しました")
    
    def update_camera(self):
        if not self.camera_paused and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # OpenCVのBGRからRGBに変換
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # PILイメージに変換
                image = Image.fromarray(frame)
                # Tkinter用に変換
                photo = ImageTk.PhotoImage(image=image)
                self.camera_label.configure(image=photo)
                self.camera_label.image = photo
        
        # 30ms後に再度更新
        self.root.after(30, self.update_camera)
    
    def capture_image(self):
        if self.camera_paused:
            self.status_label.configure(text="カメラが一時停止中です（Portrait Experience使用中）")
            return
            
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # キューが満杯かチェック
                if self.image_queue.full():
                    self.status_label.configure(text="処理待ちが満杯です。しばらくお待ちください (最大10件)")
                    return
                
                # 画像を保存
                os.makedirs('Images/in', exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                image_path = f'Images/in/captured_{timestamp}.jpg'
                cv2.imwrite(image_path, frame)
                
                # 画像をキューに追加
                self.image_queue.put(image_path)
                
                # キューの状態を表示
                queue_size = self.image_queue.qsize()
                if self.processing:
                    self.status_label.configure(text=f"画像を追加しました (キュー: {queue_size}/10件, 処理中)")
                else:
                    self.status_label.configure(text=f"画像を追加しました (キュー: {queue_size}/10件)")
    
    def select_image(self):
        """既存の画像ファイルを選択して処理"""
        file_path = filedialog.askopenfilename(
            title="画像を選択",
            filetypes=[
                ("画像ファイル", "*.jpg *.jpeg *.png *.bmp"),
                ("すべてのファイル", "*.*")
            ]
        )
        
        if file_path:
            # キューが満杯かチェック
            if self.image_queue.full():
                self.status_label.configure(text="処理待ちが満杯です。しばらくお待ちください (最大10件)")
                return
            
            # 選択された画像をImagesフォルダにコピー
            os.makedirs('Images/in', exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.basename(file_path)
            name, ext = os.path.splitext(filename)
            new_path = f'Images/in/{name}_{timestamp}{ext}'
            
            # 画像をコピー
            image = cv2.imread(file_path)
            cv2.imwrite(new_path, image)
            
            # 画像をキューに追加
            self.image_queue.put(new_path)
            
            # キューの状態を表示
            queue_size = self.image_queue.qsize()
            if self.processing:
                self.status_label.configure(text=f"画像を追加しました: {filename} (キュー: {queue_size}/10件, 処理中)")
            else:
                self.status_label.configure(text=f"画像を追加しました: {filename} (キュー: {queue_size}/10件)")
    
    def start_processing_thread(self):
        self.processing_thread = threading.Thread(target=self.process_images, daemon=True)
        self.processing_thread.start()
    
    def process_images(self):
        while True:
            if not self.image_queue.empty():
                # ロックを取得して同時処理を防ぐ
                with self.processing_lock:
                    image_path = self.image_queue.get()
                    queue_remaining = self.image_queue.qsize()
                    self.generate_video(image_path, queue_remaining)
                    self.image_queue.task_done()
            time.sleep(0.1)
    
    def generate_video(self, image_path, queue_remaining=0):
        self.processing = True
        
        try:
            self.status_label.configure(text=f"モデルロード中... (残り{queue_remaining}件)")
            self.progress['value'] = 10
            
            # モデルをロード（遅延ロード）
            self.setup_model()
            
            # 選択された出力形式を取得
            output_format = self.output_format_var.get()
            self.status_label.configure(text=f"生成中: {os.path.basename(image_path)} (残り{queue_remaining}件)")
            self.progress['value'] = 30
            
            # GPU メモリをクリア
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 画像処理
            data = self.dataset.dataset.get_item_from_path(image_path)
            self.progress['value'] = 50
            
            # 推論実行
            with torch.no_grad():  # 勾配計算を無効化してメモリ節約
                visuals = self.model.inference(data)
            
            self.progress['value'] = 70
            
            # 出力処理
            os.makedirs('Images/out', exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            generated_files = []
            
            # 動画生成
            if output_format in ["動画のみ", "動画+画像フレーム", "動画+画像まとめ", "すべて"]:
                video_path = os.path.join('Images/out', f'output_{timestamp}.mp4')
                self.visualizer.make_video(visuals, video_path)
                generated_files.append(("動画", video_path))
                
                # 生成済みファイルリストに追加
                self.generated_videos.append(video_path)
                self.generated_files.append(("動画", video_path))
                self.video_listbox.insert(tk.END, f"[動画] {os.path.basename(video_path)}")
            
            # 画像フレーム生成
            if output_format in ["画像フレームのみ", "動画+画像フレーム", "すべて"]:
                frame_dir = os.path.join('Images/out', f'frames_{timestamp}')
                self.visualizer.save_frame_images(visuals, frame_dir)
                generated_files.append(("画像フレーム", frame_dir))
                
                # 生成済みファイルリストに追加
                self.generated_files.append(("画像フレーム", frame_dir))
                self.video_listbox.insert(tk.END, f"[画像フレーム] frames_{timestamp}")
            
            # 画像まとめ生成
            if output_format in ["画像まとめのみ", "動画+画像まとめ", "すべて"]:
                image_path_output = os.path.join('Images/out', f'combined_{timestamp}.png')
                self.visualizer.save_row_image(visuals, image_path_output, traverse=True)
                generated_files.append(("画像まとめ", image_path_output))
                
                # 生成済みファイルリストに追加
                self.generated_files.append(("画像まとめ", image_path_output))
                self.video_listbox.insert(tk.END, f"[画像まとめ] {os.path.basename(image_path_output)}")
            
            # 論文用画像生成（オリジナル画像あり）
            if output_format in ["論文用画像", "動画+論文用画像", "すべて"]:
                paper_path_output = os.path.join('Images/out', f'paper_{timestamp}.png')
                self.visualizer.save_paper_image(visuals, paper_path_output, include_original=True)
                generated_files.append(("論文用画像", paper_path_output))
                
                # 生成済みファイルリストに追加
                self.generated_files.append(("論文用画像", paper_path_output))
                self.video_listbox.insert(tk.END, f"[論文用画像] {os.path.basename(paper_path_output)}")
            
            # 論文用画像生成（生成画像のみ）
            if output_format in ["論文用画像（生成のみ）", "すべて"]:
                paper_gen_path_output = os.path.join('Images/out', f'paper_gen_{timestamp}.png')
                self.visualizer.save_paper_image(visuals, paper_gen_path_output, include_original=False)
                generated_files.append(("論文用画像（生成のみ）", paper_gen_path_output))
                
                # 生成済みファイルリストに追加
                self.generated_files.append(("論文用画像（生成のみ）", paper_gen_path_output))
                self.video_listbox.insert(tk.END, f"[論文用画像（生成のみ）] {os.path.basename(paper_gen_path_output)}")
            
            # メモリ解放
            del data
            if visuals is not None:
                del visuals
            
            # GPU メモリクリアとガベージコレクション
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            gc.collect()
            
            self.progress['value'] = 90
            
            # PortraitExperienceが開いている場合、動画リストを更新
            for experience_instance in self.experience_instances[:]:  # コピーしたリストでイテレート
                try:
                    experience_instance.refresh_video_list()
                except tk.TclError:
                    # ウィンドウが閉じられている場合
                    self.experience_instances.remove(experience_instance)
            
            self.progress['value'] = 100
            
            # ステータス表示
            remaining_queue = self.image_queue.qsize()
            if len(generated_files) > 1:
                file_names = [os.path.basename(path) for _, path in generated_files]
                if remaining_queue > 0:
                    self.status_label.configure(text=f"生成完了: {', '.join(file_names)} (次の処理: {remaining_queue}件)")
                else:
                    self.status_label.configure(text=f"生成完了: {', '.join(file_names)} (処理完了)")
            else:
                file_type, file_path = generated_files[0]
                if remaining_queue > 0:
                    self.status_label.configure(text=f"{file_type}を生成しました: {os.path.basename(file_path)} (次の処理: {remaining_queue}件)")
                else:
                    self.status_label.configure(text=f"{file_type}を生成しました: {os.path.basename(file_path)} (処理完了)")
            
            # 処理完了後、モデルを解放してメモリを節約
            self.cleanup_model()
            
        except Exception as e:
            # エラー時もメモリクリア
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            gc.collect()
            self.status_label.configure(text=f"エラーが発生しました: {str(e)}")
            print(f"詳細なエラー: {e}")
            
            # エラー時もモデルを解放
            try:
                self.cleanup_model()
            except:
                pass
        finally:
            self.processing = False
    
    def start_experience(self):
        selected_indices = self.video_listbox.curselection()
        if selected_indices:
            file_type, file_path = self.generated_files[selected_indices[0]]
            
            # 動画ファイルのみ体験可能
            if file_type == "動画":
                # カメラを一時的に解放してPortrait Experienceでの競合を避ける
                self.pause_camera()
                
                # 新しいウィンドウで体験を開始
                experience_window = tk.Toplevel(self.root)
                from portraits_experience import PortraitExperience
                experience_instance = PortraitExperience(experience_window, file_path, self.on_experience_closed)
                self.experience_instances.append(experience_instance)
            elif file_type == "画像フレーム":
                self.status_label.configure(text="体験機能は動画ファイルのみに対応しています。画像フレームは「ファイルを開く」で確認できます")
            elif file_type == "論文用画像":
                self.status_label.configure(text="体験機能は動画ファイルのみに対応しています。論文用画像は「ファイルを開く」で確認できます")
            elif file_type == "論文用画像（生成のみ）":
                self.status_label.configure(text="体験機能は動画ファイルのみに対応しています。論文用画像は「ファイルを開く」で確認できます")
            else:
                self.status_label.configure(text="体験機能は動画ファイルのみに対応しています")
    
    def on_experience_closed(self, experience_instance):
        """PortraitExperienceウィンドウが閉じられたときのコールバック"""
        if experience_instance in self.experience_instances:
            self.experience_instances.remove(experience_instance)
        
        # すべてのPortrait Experienceが閉じられたらカメラを再開
        if not self.experience_instances:
            self.resume_camera()
    
    def on_background_changed(self, event):
        """背景色が変更されたときの処理"""
        selected = self.background_var.get()
        print(f"背景色が変更されました: {selected}")
        
        # 背景色の値を設定
        color_map = {"黒": 0, "グレー": 128, "白": 255}
        background_color = color_map.get(selected, 0)
        
        # データセットが既にロードされている場合は背景色を更新
        if hasattr(self, 'dataset') and self.dataset is not None:
            self.dataset.dataset.set_background_color(background_color)
            print(f"背景色を {selected}({background_color}) に設定しました")
    
    def on_gender_changed(self, event):
        """性別が変更されたときの処理"""
        selected = self.gender_var.get()
        print(f"性別が変更されました: {selected}")
        
        # 既存のモデルをクリーンアップ
        if self.model is not None:
            print("既存のモデルをクリーンアップしています...")
            self.cleanup_model()
        
        # 新しい設定でモデルをロード
        print(f"新しいモデル({selected})をロード中...")
        self.status_label.configure(text=f"モデル切り替え中: {selected}")
        
        # setup_modelが性別に応じて適切なモデルを設定するので、直接呼び出す
        try:
            self.setup_model()
            self.status_label.configure(text=f"モデル切り替え完了: {selected}")
            print(f"{selected}用のモデルロードが完了しました")
        except Exception as e:
            self.status_label.configure(text=f"モデル切り替えエラー: {str(e)}")
            print(f"モデル切り替えエラー: {e}")
    
    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()
        
        # 終了時にも念のためリソースを解放
        try:
            self.cleanup_model()
        except:
            pass

if __name__ == "__main__":
    root = tk.Tk()
    app = PortraitApp(root)
    root.mainloop() 