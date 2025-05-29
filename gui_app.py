import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import os
import time
import torch
import threading
import queue
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
        
        # モデル設定
        self.setup_model()
        
        # 画像生成キュー
        self.image_queue = queue.Queue()
        self.processing = False
        
        # 生成済み動画のリスト
        self.generated_videos = []
        
        # PortraitExperienceのインスタンス参照
        self.experience_instances = []
        
        # GUI要素の作成
        self.create_widgets()
        
        # カメラプレビューの更新
        self.update_camera()
        
        # 画像処理スレッドの開始
        self.start_processing_thread()
    
    def setup_model(self):
        # モデルの初期化
        opt = TestOptions().parse(save=False)
        opt.display_id = 0
        opt.gpu_ids = [0]
        opt.nThreads = 1
        opt.batchSize = 1
        opt.serial_batches = True
        opt.no_flip = True
        opt.in_the_wild = True
        opt.traverse = True
        opt.interp_step = 0.05
        opt.no_moving_avg = True
        opt.fineSize = 256
        
        self.data_loader = CreateDataLoader(opt)
        self.dataset = self.data_loader.load_data()
        self.visualizer = Visualizer(opt)
        
        opt.name = 'males_model'
        self.model = create_model(opt)
        self.model.eval()
        
        # GPUメモリ最適化
        torch.cuda.empty_cache()
    
    def create_widgets(self):
        # メインフレーム
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # カメラプレビュー
        self.camera_label = ttk.Label(main_frame)
        self.camera_label.grid(row=0, column=0, columnspan=2, padx=5, pady=5)
        
        # 撮影ボタン
        self.capture_btn = ttk.Button(main_frame, text="撮影", command=self.capture_image)
        self.capture_btn.grid(row=1, column=0, padx=5, pady=5)
        
        # 生成済み動画リスト
        self.video_listbox = tk.Listbox(main_frame, height=5)
        self.video_listbox.grid(row=2, column=0, columnspan=2, padx=5, pady=5)
        
        # 進捗バー
        self.progress = ttk.Progressbar(main_frame, length=200, mode='determinate')
        self.progress.grid(row=3, column=0, columnspan=2, padx=5, pady=5)
        
        # ステータスラベル
        self.status_label = ttk.Label(main_frame, text="準備完了")
        self.status_label.grid(row=4, column=0, columnspan=2, padx=5, pady=5)
        
        # 体験開始ボタン
        self.experience_btn = ttk.Button(main_frame, text="体験開始", command=self.start_experience)
        self.experience_btn.grid(row=5, column=0, columnspan=2, padx=5, pady=5)
    
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
                # 画像を保存
                os.makedirs('Images/in', exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                image_path = f'Images/in/captured_{timestamp}.jpg'
                cv2.imwrite(image_path, frame)
                self.status_label.configure(text=f"画像を保存しました: {image_path}")
                
                # 画像をキューに追加
                self.image_queue.put(image_path)
    
    def start_processing_thread(self):
        self.processing_thread = threading.Thread(target=self.process_images, daemon=True)
        self.processing_thread.start()
    
    def process_images(self):
        while True:
            if not self.image_queue.empty():
                image_path = self.image_queue.get()
                self.generate_video(image_path)
                self.image_queue.task_done()
            time.sleep(0.1)
    
    def generate_video(self, image_path):
        self.status_label.configure(text=f"動画生成中: {image_path}")
        self.progress['value'] = 0
        
        try:
            # 画像処理
            data = self.dataset.dataset.get_item_from_path(image_path)
            
            # 推論実行
            visuals = self.model.inference(data)
            
            # 出力処理
            os.makedirs('Images/out', exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join('Images/out', f'output_{timestamp}.mp4')
            self.visualizer.make_video(visuals, out_path)
            
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
            self.status_label.configure(text=f"動画を生成しました: {out_path}")
        except Exception as e:
            self.status_label.configure(text=f"エラーが発生しました: {str(e)}")
    
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
    
    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = PortraitApp(root)
    root.mainloop() 