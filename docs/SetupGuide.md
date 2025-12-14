# MoonInTheRiver版 DiffSinger 環境構築ガイド（uv + Windows）

このガイドでは、MoonInTheRiver版DiffSingerをWindows環境でuvを使用してセットアップし、推論を実行する手順を説明します。

## 概要

### MoonInTheRiver版とOpenVPI版の違い

| 項目 | MoonInTheRiver版 | OpenVPI版 |
|------|------------------|-----------|
| サンプルレート | 24kHz | 44.1kHz |
| Mel bins | 80 | 128 |
| Hop size | 128 | 512 |
| 公式デモモデル | あり | なし |
| フレームワーク | PyTorch Lightning (旧) | PyTorch Lightning (新) |
| 互換性 | 両者で**互換性なし** | 両者で**互換性なし** |

### 本ガイドで使用するモデル

- **アコースティックモデル**: 0831_opencpop_ds1000（中国語歌声合成）
- **ボコーダー**: 0109_hifigan_bigpopcs_hop128
- **ピッチ抽出器**: 0102_xiaoma_pe

---

## 前提条件

- **Python**: 3.10〜3.12（PyTorchがPython 3.13をサポートしていないため）
- **uv**: [https://docs.astral.sh/uv/](https://docs.astral.sh/uv/) がインストール済み
- **GPU（オプション）**: NVIDIA GPU + CUDA 12.x
- **OS**: Windows 10/11（本ガイドはWindows向けの修正を含む）

---

## 環境構築手順

### 1. リポジトリのクローン

```bash
cd C:/Users/yuta/Desktop/Titan
git clone https://github.com/MoonInTheRiver/DiffSinger.git DiffSinger-MoonInTheRiver
cd DiffSinger-MoonInTheRiver
```

### 2. uv プロジェクトの初期化

```bash
uv init --name diffsinger-moonintheriver --no-readme
```

### 3. pyproject.toml の編集

`pyproject.toml` を以下のように編集します：

```toml
[project]
name = "diffsinger-moonintheriver"
version = "0.1.0"
description = "Add your description here"
requires-python = ">=3.10,<3.13"
dependencies = []

[[tool.uv.index]]
url = "https://download.pytorch.org/whl/cu121"
name = "pytorch-cu121"
```

> **注意**: CPU版を使用する場合は `[[tool.uv.index]]` セクションを削除してください。

### 4. Python バージョンの固定

```bash
uv python pin 3.10
```

### 5. PyTorch のインストール

```bash
uv add torch torchvision torchaudio
```

### 6. コア依存関係のインストール

```bash
uv add "numpy<2.0.0" "scipy>=1.3.0" "librosa>=0.8.0,<0.10.0" matplotlib tqdm pandas PyYAML tensorboard tensorboardX einops h5py pyworld "praat-parselmouth>=0.3.3" resampy soundfile
```

### 7. テキスト処理用依存関係

```bash
uv add g2p_en pypinyin jieba pyloudnorm
```

### 8. PyTorch Lightning

```bash
uv add "pytorch-lightning>=1.0.0,<2.0.0"
```

### 9. 追加の依存関係

```bash
uv add pycwt scikit-image
uv pip install webrtcvad-wheels
```

> **注意**: `webrtcvad` は C++ コンパイラが必要なため、プリビルドの `webrtcvad-wheels` を使用します。

### 10. 動作確認

```bash
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

期待される出力：
```
PyTorch: 2.5.1+cu121
CUDA available: True
```

---

## Windows 固有の修正

Windows環境では以下の3つの修正が必要です。

### 修正1: scipy.signal.kaiser

新しいscipyでは `kaiser` 関数の場所が変更されています。

**ファイル**: `modules/parallel_wavegan/layers/pqmf.py`

```python
# 修正前
from scipy.signal import kaiser

# 修正後
from scipy.signal.windows import kaiser
```

### 修正2: YAML ファイルの UTF-8 エンコーディング

**ファイル**: `utils/hparams.py`

```python
# 修正前（2箇所）
with open(config_fn) as f:

# 修正後
with open(config_fn, encoding='utf-8') as f:
```

### 修正3: Windows パスの正規表現

**ファイル**: `utils/__init__.py` (load_ckpt関数内)

```python
# 修正前
lambda x: int(re.findall(f'{base_dir}/model_ckpt_steps_(\d+).ckpt', x)[0])

# 修正後
lambda x: int(re.findall(r'model_ckpt_steps_(\d+)\.ckpt', x)[0])
```

**ファイル**: `inference/svs/base_svs_infer.py` (build_vocoder関数内)

```python
# 修正前
lambda x: int(re.findall(f'{base_dir}/model_ckpt_steps_(\d+).ckpt', x)[0])

# 修正後
lambda x: int(re.findall(r'model_ckpt_steps_(\d+)\.ckpt', x)[0])
```

---

## モデルのダウンロードと配置

### ダウンロードURL

| モデル | URL |
|--------|-----|
| ボコーダー | https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/0109_hifigan_bigpopcs_hop128.zip |
| アコースティック | https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/0831_opencpop_ds1000.zip |
| ピッチ抽出器 | https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/0102_xiaoma_pe.zip |

### ダウンロードと配置

```bash
cd checkpoints

# ボコーダー
curl -L -o hifigan.zip "https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/0109_hifigan_bigpopcs_hop128.zip"
mkdir 0109_hifigan_bigpopcs_hop128
unzip hifigan.zip -d 0109_hifigan_bigpopcs_hop128/
rm hifigan.zip

# アコースティックモデル
curl -L -o opencpop.zip "https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/0831_opencpop_ds1000.zip"
mkdir 0831_opencpop_ds1000
unzip opencpop.zip -d 0831_opencpop_ds1000/
rm opencpop.zip

# ピッチ抽出器
curl -L -o xiaoma_pe.zip "https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/0102_xiaoma_pe.zip"
mkdir 0102_xiaoma_pe
unzip xiaoma_pe.zip -d 0102_xiaoma_pe/
rm xiaoma_pe.zip
```

### 期待されるディレクトリ構造

```
checkpoints/
├── 0109_hifigan_bigpopcs_hop128/
│   ├── config.yaml
│   └── model_ckpt_steps_280000.ckpt
├── 0831_opencpop_ds1000/
│   ├── config.yaml
│   └── model_ckpt_steps_320000.ckpt
└── 0102_xiaoma_pe/
    ├── config.yaml
    └── model_ckpt_steps_60000.ckpt
```

---

## 推論実行

### コマンド

```bash
cd DiffSinger-MoonInTheRiver
PYTHONPATH=. uv run python inference/svs/ds_e2e.py --config checkpoints/0831_opencpop_ds1000/config.yaml --exp_name 0831_opencpop_ds1000
```

### 入力形式

推論スクリプト (`inference/svs/ds_e2e.py`) 内の `inp` を編集して入力を変更できます：

#### 単語レベル入力

```python
inp = {
    'text': '小酒窝长睫毛AP是你最美的记号',
    'notes': 'C#4/Db4 | F#4/Gb4 | G#4/Ab4 | A#4/Bb4 F#4/Gb4 | F#4/Gb4 C#4/Db4 | C#4/Db4 | rest | C#4/Db4 | A#4/Bb4 | G#4/Ab4 | A#4/Bb4 | G#4/Ab4 | F4 | C#4/Db4',
    'notes_duration': '0.407140 | 0.376190 | 0.242180 | 0.509550 0.183420 | 0.315400 0.235020 | 0.361660 | 0.223070 | 0.377270 | 0.340550 | 0.299620 | 0.344510 | 0.283770 | 0.323390 | 0.360340',
    'input_type': 'word'
}
```

#### 音素レベル入力

```python
inp = {
    'text': '小酒窝长睫毛AP是你最美的记号',
    'ph_seq': 'x iao j iu w o ch ang ang j ie ie m ao AP sh i n i z ui m ei d e j i h ao',
    'note_seq': 'C#4/Db4 C#4/Db4 F#4/Gb4 F#4/Gb4 ...',
    'note_dur_seq': '0.407140 0.407140 0.376190 0.376190 ...',
    'is_slur_seq': '0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0',
    'input_type': 'phoneme'
}
```

### 出力

生成された音声ファイルは `infer_out/example_out.wav` に保存されます。

---

## トラブルシューティング

### エラー: `ModuleNotFoundError: No module named 'xxx'`

不足しているパッケージをインストールします：

```bash
uv add <package_name>
# または
uv pip install <package_name>
```

### エラー: `ImportError: cannot import name 'kaiser' from 'scipy.signal'`

[修正1](#修正1-scipysignalkaiser) を適用してください。

### エラー: `UnicodeDecodeError: 'cp932' codec can't decode byte`

[修正2](#修正2-yaml-ファイルの-utf-8-エンコーディング) を適用してください。

### エラー: `IndexError: list index out of range` (load_ckpt)

[修正3](#修正3-windows-パスの正規表現) を適用してください。

### エラー: `KeyError: 'f0_denorm'`

`ds_cascade.py` ではなく `ds_e2e.py` を使用してください。0831_opencpop_ds1000 モデルはピッチ抽出器（PE）が有効なため、e2e推論スクリプトが必要です。

---

## 参考リンク

- [MoonInTheRiver/DiffSinger](https://github.com/MoonInTheRiver/DiffSinger)
- [OpenVPI/DiffSinger](https://github.com/openvpi/DiffSinger)（別バージョン）
- [uv ドキュメント](https://docs.astral.sh/uv/)
