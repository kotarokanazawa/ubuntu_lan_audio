# Voice Bridge (Opus UDP Audio)

軽量な双方向音声ブリッジです。UDPで音声を送受信し、Opusで圧縮します。GUIはシンプル表示と詳細設定に分かれています。

---

## 特徴

* 双方向リアルタイム音声通信（UDP）
* Opusによる低ビットレート音声圧縮
* VAD（無音検出）による帯域削減
* AEC（エコーキャンセル）対応（システム機能使用）
* GUIによるリアルタイム調整
* YAMLで設定保存・読み込み

---

## 起動

```bash
python3 audio_cross_opus_simple_mode_v3.py
```

またはCLIモード

```bash
python3 audio_cross_opus_simple_mode_v3.py --peer-host 192.168.0.10
```

---

## 必要環境

* Python 3.9+
* Linux（PulseAudio または PipeWire）

### 依存ライブラリ

```bash
pip install sounddevice numpy
```

```bash
sudo apt install libopus0 libopus-dev
```

AEC使用時

```bash
sudo apt install pulseaudio-utils
```

---

## 基本構成

2台で起動することで通話できます。

例：

### PC A

```bash
--listen-port 5000 --peer-host PC_B_IP --peer-port 5001
```

### PC B

```bash
--listen-port 5001 --peer-host PC_A_IP --peer-port 5000
```

---

## GUI（シンプルモード）

通常画面は最小構成です。

### Session

* IP
* 送信ポート
* 受信ポート

### Devices

* 入力デバイス
* 出力デバイス

### Volume

* 入力ゲイン
* 出力ゲイン

### Levels

* マイク入力レベル
* 受信音声レベル

### 上部ボタン

* Connect / Disconnect
* Ping（疎通確認）
* Advanced（詳細設定）
* Load / Save（設定ファイル）

---

## Advanced（詳細設定）

以下の調整が可能です。

* Bitrate（音質 / 帯域）
* Frameサイズ
* Complexity（CPU使用量）
* VAD関連
* フィルタ / ゲート / リミッタ
* AEC方式（webrtc / speex）
* Queueサイズ（遅延）

---

## 設定ファイル

保存

```bash
--save-config config.yaml
```

読み込み

```bash
--config config.yaml
```

---

## 通信テスト

```bash
--test-connection
```

RTTがログに表示されます。

---

## 注意

* 同一ネットワーク内での使用を前提
* NAT越えは未対応
* AECはシステム依存（環境によって品質差あり）

---

## ライセンス

個人利用・改変自由
