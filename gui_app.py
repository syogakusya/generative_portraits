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
        self.cap = cv2.VideoCapture(0)
        self.is_capturing = False
        
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
        
        self.data_loader = CreateDataLoader(self.opt)
        self.dataset = self.data_loader.load_data()
        self.visualizer = Visualizer(self.opt)
        
        # 現在選択されている背景色を適用
        selected = self.background_var.get()
        color_map = {"黒": 0, "グレー": 128, "白": 255}
        background_color = color_map.get(selected, 0)
        self.dataset.dataset.set_background_color(background_color)
        
        self.opt.name = 'males_model'
        self.model = create_model(self.opt)
        self.model.eval()
        
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
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # カメラプレビュー
        self.camera_label = ttk.Label(main_frame)
        self.camera_label.grid(row=0, column=0, columnspan=2, padx=5, pady=5)
        
        # ボタンフレーム
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5)
        
        # 撮影ボタン
        self.capture_btn = ttk.Button(button_frame, text="撮影", command=self.capture_image)
        self.capture_btn.grid(row=0, column=0, padx=5)
        
        # 画像選択ボタン
        self.select_image_btn = ttk.Button(button_frame, text="画像を選択", command=self.select_image)
        self.select_image_btn.grid(row=0, column=1, padx=5)
        
        # 生成済み動画リスト
        list_frame = ttk.Frame(main_frame)
        list_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # リストボックスのラベル
        self.list_label = ttk.Label(list_frame, text="生成済み動画：")
        self.list_label.grid(row=0, column=0, sticky=tk.W)
        
        # スクロールバー付きリストボックス
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self.video_listbox = tk.Listbox(list_frame, height=5, yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.video_listbox.yview)
        
        self.video_listbox.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=1, column=1, sticky=(tk.N, tk.S))
        
        # 既存の動画を表示
        for video_path in self.generated_videos:
            self.video_listbox.insert(tk.END, os.path.basename(video_path))
        
        # 進捗バー
        self.progress = ttk.Progressbar(main_frame, length=200, mode='determinate')
        self.progress.grid(row=3, column=0, columnspan=2, padx=5, pady=5)
        
        # ステータスラベル
        self.status_label = ttk.Label(main_frame, text="準備完了")
        self.status_label.grid(row=4, column=0, columnspan=2, padx=5, pady=5)
        
        # 体験開始ボタン
        self.experience_btn = ttk.Button(main_frame, text="体験開始", command=self.start_experience)
        self.experience_btn.grid(row=5, column=0, columnspan=2, padx=5, pady=5)
        
        # 背景色選択
        background_frame = ttk.Frame(main_frame)
        background_frame.grid(row=6, column=0, columnspan=2, padx=5, pady=5)
        
        ttk.Label(background_frame, text="背景色:").grid(row=0, column=0, padx=5)
        self.background_var = tk.StringVar(value="黒")
        self.background_combo = ttk.Combobox(background_frame, textvariable=self.background_var, 
                                           values=["黒", "グレー", "白"], state="readonly")
        self.background_combo.grid(row=0, column=1, padx=5)
        self.background_combo.bind('<<ComboboxSelected>>', self.on_background_changed)
        
        # メモリクリアボタン（デバッグ用）
        self.cleanup_btn = ttk.Button(main_frame, text="メモリクリア", command=self.manual_cleanup)
        self.cleanup_btn.grid(row=7, column=0, columnspan=2, padx=5, pady=5)
    
    def manual_cleanup(self):
        """手動でメモリクリアを実行"""
        self.cleanup_model()
        self.status_label.configure(text="メモリクリアを実行しました")
    
    def update_camera(self):
        if self.cap.isOpened():
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
            
            self.status_label.configure(text=f"動画生成中: {os.path.basename(image_path)} (残り{queue_remaining}件)")
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
            out_path = os.path.join('Images/out', f'output_{timestamp}.mp4')
            self.visualizer.make_video(visuals, out_path)
            
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
            
            # 生成済み動画リストに追加
            self.generated_videos.append(out_path)
            self.video_listbox.insert(tk.END, os.path.basename(out_path))
            
            # PortraitExperienceが開いている場合、動画リストを更新
            for experience_instance in self.experience_instances[:]:  # コピーしたリストでイテレート
                try:
                    experience_instance.refresh_video_list()
                except tk.TclError:
                    # ウィンドウが閉じられている場合
                    self.experience_instances.remove(experience_instance)
            
            self.progress['value'] = 100
            
            # 次のキューの状態をチェック
            remaining_queue = self.image_queue.qsize()
            if remaining_queue > 0:
                self.status_label.configure(text=f"動画を生成しました: {os.path.basename(out_path)} (次の処理: {remaining_queue}件)")
            else:
                self.status_label.configure(text=f"動画を生成しました: {os.path.basename(out_path)} (処理完了)")
            
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
            selected_video = self.generated_videos[selected_indices[0]]
            # 新しいウィンドウで体験を開始
            experience_window = tk.Toplevel(self.root)
            from portraits_experience import PortraitExperience
            experience_instance = PortraitExperience(experience_window, selected_video, self.on_experience_closed)
            self.experience_instances.append(experience_instance)
    
    def on_experience_closed(self, experience_instance):
        """PortraitExperienceウィンドウが閉じられたときのコールバック"""
        if experience_instance in self.experience_instances:
            self.experience_instances.remove(experience_instance)
    
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