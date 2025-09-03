# 注意
このリポジトリのプログラムはCUDAに対応しているOSが必要なため、macOSでは動作しません。

# プログラム実行手順
## generative_portraits内で以下を実行してください
```bash
.\venv\Scripts\activate

python .\gui_app.py
```
# クローン後のセットアップ手順（venv推奨）

このリポジトリをクローンした直後に行うべきセットアップ手順をまとめます。

---
## 1. Python仮想環境（venv）の作成と有効化
```bash
python3 -m venv venv

# Linuxの場合
source venv/bin/activate

# Windowsの場合
.\venv\Scripts\activate
```

## 2. 依存パッケージのインストール
```bash
pip install -r requirements.txt
```

## 3. PyTorchとtorchvisionのインストール（必要に応じて）
- CUDA対応GPUをお持ちの場合は[PyTorch公式サイト](https://pytorch.org/)の指示に従ってインストールしてください。
- 例（CUDA 11.8の場合）:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

## 4. 学習済みモデルのダウンロード
```bash
python download_models.py
```

## 5. 動作確認
- デモやテストの手順は元の[README](./README_Source.md)内「Quick Demo」や「Testing」セクションを参照してください。

---

### 備考
- venv環境を有効化すると、プロンプトの先頭に`(venv)`と表示されます。
- 新しいターミナルを開くたびに、venv環境を有効化する必要があります。
- 依存パッケージは`requirements.txt`に記載されています。
- PyTorchのバージョンやCUDAバージョンはご自身の環境に合わせて調整してください。 

---

