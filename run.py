import os, shutil
import time
import torch
from collections import OrderedDict
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer

# 全体の実行時間計測開始
total_start_time = time.time()

# CUDA高速化設定
torch.backends.cudnn.benchmark = True

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

# データローダーの初期化時間計測
loader_start_time = time.time()
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
loader_end_time = time.time()
print(f'データローダー初期化時間: {loader_end_time - loader_start_time:.2f}秒')

visualizer = Visualizer(opt)

# モデルの初期化時間計測
model_start_time = time.time()
opt.name = 'males_model'
model = create_model(opt)
model.eval()

# GPUメモリ最適化
torch.cuda.empty_cache()

model_end_time = time.time()
print(f'モデル初期化時間: {model_end_time - model_start_time:.2f}秒')

# 画像処理時間計測
img_path = "Images/in/syoumeu.jpg"
print(f'Using local file "{img_path}"')

inference_start_time = time.time()
data = dataset.dataset.get_item_from_path(img_path)

# 推論実行
visuals = model.inference(data)

inference_end_time = time.time()
print(f'推論時間: {inference_end_time - inference_start_time:.2f}秒')

# 出力処理時間計測
output_start_time = time.time()
shutil.rmtree('Images/out')
os.makedirs('Images/out', exist_ok=True)

filename = os.path.basename(img_path)
out_path = os.path.join('Images/out', os.path.splitext(filename)[0] + '.mp4')
visualizer.make_video(visuals, out_path)
output_end_time = time.time()
print(f'出力処理時間: {output_end_time - output_start_time:.2f}秒')

# 全体の実行時間計測終了
total_end_time = time.time()
print(f'全体の実行時間: {total_end_time - total_start_time:.2f}秒')

print(f'Generated output to "{out_path}"')