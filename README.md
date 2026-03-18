# Codec2 Audio Cross Bridge

Ubuntu/Linux 向けの、双方向マイク/スピーカー交換用のオフライン音声ツールです。

- GUI あり
- CLI あり
- UDP で双方向通信
- Codec2 による低帯域化
- VAD/DTX により無音時は送信停止
- 低遅延寄りの小フレーム動作
- 入力デバイスと出力デバイスを個別に指定可能
- デバイス ID でもデバイス名でも指定可能

## 特徴

主な調整項目:
- `--codec-mode`
  - `3200`: 劣化が少ないが帯域は多め
  - `2400`
  - `1600`
  - `1400`
  - `1300`
  - `1200`: かなり劣化するが帯域は最小側
- `--vad-threshold`
  - 高いほど無音判定が強くなり、平均通信量は減る
  - ただし小声や語頭が欠けやすくなる
- `--recv-queue-frames`
  - 小さいほど低遅延
  - 小さすぎると途切れやすい
- `--input-device-name` / `--output-device-name`
  - 部分一致でも選択可能
  - 複数候補に一致した場合はエラー

## 依存関係

```bash
sudo apt update
sudo apt install -y python3-tk python3-pip portaudio19-dev libcodec2-1.2 libportaudio2
pip install numpy sounddevice
```

## デバイス一覧

```bash
python3 audio_cross_codec2.py --list-devices
```

## CLI 実行例

### デバイス名で指定する例

```bash
python3 audio_cross_codec2.py \
  --listen-port 5000 \
  --peer-host 192.168.1.20 \
  --peer-port 5001 \
  --codec-mode 1300 \
  --input-device-name "USB" \
  --output-device-name "Headphones"
```

### デバイス ID で指定する例

```bash
python3 audio_cross_codec2.py \
  --listen-port 5000 \
  --peer-host 192.168.1.20 \
  --peer-port 5001 \
  --codec-mode 1300 \
  --input-device 2 \
  --output-device 5
```

優先順位は次の通りです。

1. `--input-device` / `--output-device`
2. `--input-device-name` / `--output-device-name`
3. どちらも未指定ならシステム既定デバイス

## GUI 実行

```bash
python3 audio_cross_codec2.py --gui
```

引数なしでも GUI が開きます。

```bash
python3 audio_cross_codec2.py
```

GUI では次のことができます。

- 入力デバイス名と出力デバイス名を別々に選択
- 必要なら入力 ID と出力 ID を別々に手入力
- `Refresh devices` で一覧更新
- `Use default` で個別に既定デバイスへ戻す

## 遠隔操作向けの推奨初期値

```text
codec-mode = 1300
vad-threshold = 700
recv-queue-frames = 4
```

さらに帯域を詰めたい場合:

```text
codec-mode = 1200
vad-threshold = 900〜1200
recv-queue-frames = 3
```

## 制約

- Narrowband Codec2 のため `samplerate=8000` 固定運用を前提
- 暗号化なし
- NAT 越えなし
- エコーキャンセルなし
- 雑音抑制なし
- LAN / 専用ネットワーク向け
