#!/usr/bin/env python3
"""
Low-bandwidth bidirectional audio bridge for Ubuntu/Linux using Codec2.

Features
- Full-duplex mic<->speaker exchange over UDP
- Codec2 voice compression with selectable modes
- Low-latency small-frame streaming
- VAD/DTX (do not send silence) to reduce average bandwidth
- Tkinter GUI and CLI in a single file
- Separate input/output device selection by ID or device name
- YAML save/load for configuration
- Communication test and richer connection state display
- Works offline on a LAN
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import os
import queue
import socket
import struct
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import sounddevice as sd
except Exception as e:  # pragma: no cover
    sd = None
    _SOUNDDEVICE_IMPORT_ERROR = e
else:
    _SOUNDDEVICE_IMPORT_ERROR = None

try:  # optional
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

# =========================
# Codec2 binding
# =========================

_CODEC2_LIB = ctypes.util.find_library("codec2")
if not _CODEC2_LIB:
    raise RuntimeError(
        "libcodec2 not found. Install Ubuntu package: sudo apt install libcodec2-1.2"
    )

_libcodec2 = ctypes.CDLL(_CODEC2_LIB)

CODEC2_MODES: Dict[str, int] = {
    "3200": 0,
    "2400": 1,
    "1600": 2,
    "1400": 3,
    "1300": 4,
    "1200": 5,
}

_libcodec2.codec2_create.argtypes = [ctypes.c_int]
_libcodec2.codec2_create.restype = ctypes.c_void_p
_libcodec2.codec2_destroy.argtypes = [ctypes.c_void_p]
_libcodec2.codec2_destroy.restype = None
_libcodec2.codec2_bits_per_frame.argtypes = [ctypes.c_void_p]
_libcodec2.codec2_bits_per_frame.restype = ctypes.c_int
_libcodec2.codec2_samples_per_frame.argtypes = [ctypes.c_void_p]
_libcodec2.codec2_samples_per_frame.restype = ctypes.c_int
_libcodec2.codec2_encode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.POINTER(ctypes.c_short)]
_libcodec2.codec2_encode.restype = None
_libcodec2.codec2_decode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_short), ctypes.POINTER(ctypes.c_ubyte)]
_libcodec2.codec2_decode.restype = None


class Codec2:
    def __init__(self, mode_name: str):
        if mode_name not in CODEC2_MODES:
            raise ValueError(f"Unsupported Codec2 mode: {mode_name}")
        self.mode_name = mode_name
        self.mode_id = CODEC2_MODES[mode_name]
        self.handle = _libcodec2.codec2_create(self.mode_id)
        if not self.handle:
            raise RuntimeError(f"codec2_create failed for mode {mode_name}")
        self.bits_per_frame = int(_libcodec2.codec2_bits_per_frame(self.handle))
        self.bytes_per_frame = (self.bits_per_frame + 7) // 8
        self.samples_per_frame = int(_libcodec2.codec2_samples_per_frame(self.handle))

    def close(self) -> None:
        if self.handle:
            _libcodec2.codec2_destroy(self.handle)
            self.handle = None

    def encode(self, pcm_i16: np.ndarray) -> bytes:
        if pcm_i16.dtype != np.int16:
            pcm_i16 = pcm_i16.astype(np.int16, copy=False)
        if pcm_i16.ndim != 1 or len(pcm_i16) != self.samples_per_frame:
            raise ValueError(f"Codec2 encode expects {self.samples_per_frame} mono int16 samples")
        out = (ctypes.c_ubyte * self.bytes_per_frame)()
        inp = pcm_i16.ctypes.data_as(ctypes.POINTER(ctypes.c_short))
        _libcodec2.codec2_encode(self.handle, out, inp)
        return bytes(out)

    def decode(self, payload: bytes) -> np.ndarray:
        if len(payload) != self.bytes_per_frame:
            raise ValueError(f"Codec2 decode expects {self.bytes_per_frame} bytes, got {len(payload)}")
        inbuf = (ctypes.c_ubyte * self.bytes_per_frame).from_buffer_copy(payload)
        out = np.zeros(self.samples_per_frame, dtype=np.int16)
        out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_short))
        _libcodec2.codec2_decode(self.handle, out_ptr, inbuf)
        return out

    def __del__(self):  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass


# =========================
# Protocol
# =========================

MAGIC = b"C2A1"
PROTO_VERSION = 1
PKT_AUDIO = 1
PKT_PING = 2
PKT_PONG = 3
PKT_TEST = 4
PKT_TEST_REPLY = 5
HEADER_STRUCT = struct.Struct("!4sBBBBI")
MODE_ID_TO_NAME = {v: k for k, v in CODEC2_MODES.items()}


def build_packet(pkt_type: int, mode_id: int, flags: int, seq: int, payload: bytes) -> bytes:
    return HEADER_STRUCT.pack(MAGIC, PROTO_VERSION, pkt_type, mode_id, flags, seq) + payload


def parse_packet(data: bytes):
    if len(data) < HEADER_STRUCT.size:
        return None
    magic, ver, pkt_type, mode_id, flags, seq = HEADER_STRUCT.unpack_from(data)
    if magic != MAGIC or ver != PROTO_VERSION:
        return None
    return pkt_type, mode_id, flags, seq, data[HEADER_STRUCT.size:]


# =========================
# Device helpers
# =========================


def ensure_sounddevice() -> None:
    if sd is None:
        raise RuntimeError(
            "sounddevice import failed. Install it with: pip install sounddevice"
            f"\nOriginal error: {_SOUNDDEVICE_IMPORT_ERROR}"
        )


def query_audio_devices() -> List[Dict[str, Any]]:
    ensure_sounddevice()
    return [dict(d) for d in sd.query_devices()]


def list_audio_devices() -> str:
    try:
        devs = query_audio_devices()
    except Exception as e:
        return f"Device query failed: {e}"

    default_in = None
    default_out = None
    try:
        default_in, default_out = sd.default.device
    except Exception:
        pass

    lines: List[str] = []
    for i, dev in enumerate(devs):
        tags = []
        if i == default_in:
            tags.append("default-in")
        if i == default_out:
            tags.append("default-out")
        tag_txt = f" [{' '.join(tags)}]" if tags else ""
        lines.append(
            f"[{i}] {dev['name']} in={dev['max_input_channels']} out={dev['max_output_channels']}{tag_txt}"
        )
    return "\n".join(lines)


def _normalize_name(s: str) -> str:
    return " ".join(s.casefold().split())


def resolve_device(direction: str, device_id: Optional[int], device_name: Optional[str]) -> Optional[int]:
    if device_name:
        devs = query_audio_devices()
        wanted = _normalize_name(device_name)
        key = "max_input_channels" if direction == "input" else "max_output_channels"
        exact: List[Tuple[int, Dict[str, Any]]] = []
        partial: List[Tuple[int, Dict[str, Any]]] = []
        for idx, dev in enumerate(devs):
            if int(dev.get(key, 0)) <= 0:
                continue
            current = _normalize_name(str(dev.get("name", "")))
            if current == wanted:
                exact.append((idx, dev))
            elif wanted in current:
                partial.append((idx, dev))
        matches = exact or partial
        if not matches:
            raise ValueError(f"No {direction} device matched name: {device_name}")
        if len(matches) > 1:
            sample = ", ".join(f"[{idx}] {dev['name']}" for idx, dev in matches[:8])
            raise ValueError(f"Ambiguous {direction} device name '{device_name}'. Matches: {sample}")
        return matches[0][0]
    return device_id


def describe_device(device_id: Optional[int]) -> str:
    if device_id is None:
        return "default"
    try:
        devs = query_audio_devices()
        if 0 <= device_id < len(devs):
            return f"[{device_id}] {devs[device_id]['name']}"
    except Exception:
        pass
    return str(device_id)


# =========================
# YAML helpers
# =========================

CONFIG_KEYS = {
    "peer_host",
    "peer_port",
    "listen_port",
    "codec_mode",
    "samplerate",
    "input_device",
    "output_device",
    "input_device_name",
    "output_device_name",
    "vad_threshold",
    "vad_hangover_frames",
    "preemphasis",
    "input_gain",
    "recv_queue_frames",
    "bind_host",
    "ttl_ping_s",
    "ping_interval_s",
}


def _parse_scalar(value: str) -> Any:
    v = value.strip()
    if v in {"", "null", "Null", "NULL", "~"}:
        return None
    if v in {"true", "True", "TRUE"}:
        return True
    if v in {"false", "False", "FALSE"}:
        return False
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        return v[1:-1]
    try:
        if "." in v or "e" in v.lower():
            return float(v)
        return int(v)
    except Exception:
        return v


def _simple_yaml_load(path: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        out[key.strip()] = _parse_scalar(value)
    return out


def load_config_yaml(path: str) -> Dict[str, Any]:
    if yaml is not None:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError("Config YAML must be a mapping")
        return {k: data.get(k) for k in CONFIG_KEYS if k in data}
    return {k: v for k, v in _simple_yaml_load(path).items() if k in CONFIG_KEYS}


def _simple_yaml_dump(data: Dict[str, Any]) -> str:
    def fmt(v: Any) -> str:
        if v is None:
            return "null"
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return str(v)
        s = str(v)
        if s == "" or ":" in s or "#" in s or s.strip() != s or " " in s:
            return '"' + s.replace('"', '\\"') + '"'
        return s

    lines = ["# audio_cross_codec2 configuration"]
    for key in sorted(data.keys()):
        lines.append(f"{key}: {fmt(data[key])}")
    return "\n".join(lines) + "\n"


def save_config_yaml(path: str, data: Dict[str, Any]) -> None:
    data = {k: data.get(k) for k in CONFIG_KEYS}
    if yaml is not None:
        Path(path).write_text(yaml.safe_dump(data, sort_keys=True, allow_unicode=True), encoding="utf-8")
    else:
        Path(path).write_text(_simple_yaml_dump(data), encoding="utf-8")


# =========================
# Config and stats
# =========================

@dataclass
class AudioConfig:
    peer_host: str
    peer_port: int
    listen_port: int
    codec_mode: str = "1300"
    samplerate: int = 8000
    input_device: Optional[int] = None
    output_device: Optional[int] = None
    input_device_name: Optional[str] = None
    output_device_name: Optional[str] = None
    vad_threshold: int = 700
    vad_hangover_frames: int = 2
    preemphasis: float = 0.0
    input_gain: float = 1.0
    recv_queue_frames: int = 4
    bind_host: str = "0.0.0.0"
    ttl_ping_s: float = 1.0
    ping_interval_s: float = 0.5


@dataclass
class AudioStats:
    tx_packets: int = 0
    tx_bytes_udp_payload: int = 0
    rx_packets: int = 0
    rx_bytes_udp_payload: int = 0
    tx_voice_frames: int = 0
    tx_dtx_drops: int = 0
    rx_silence_fill_frames: int = 0
    last_rx_seq: Optional[int] = None
    rx_seq_gaps: int = 0
    started_at: float = field(default_factory=time.time)
    last_pong_at: float = 0.0
    last_test_reply_at: float = 0.0
    last_test_rtt_ms: Optional[float] = None
    test_requests: int = 0
    test_replies: int = 0

    def avg_kbps(self) -> float:
        dt = max(0.001, time.time() - self.started_at)
        return (self.tx_bytes_udp_payload * 8.0) / dt / 1000.0


# =========================
# Engine
# =========================

class AudioBridge:
    def __init__(self, config: AudioConfig, logger: Optional[Callable[[str], None]] = None):
        ensure_sounddevice()
        config.input_device = resolve_device("input", config.input_device, config.input_device_name)
        config.output_device = resolve_device("output", config.output_device, config.output_device_name)
        self.cfg = config
        self.log = logger or (lambda msg: print(msg, flush=True))
        self.tx_codec = Codec2(config.codec_mode)
        self.rx_codecs: Dict[int, Codec2] = {}
        self.frame_samples = self.tx_codec.samples_per_frame
        if self.cfg.samplerate != 8000:
            raise ValueError("Codec2 narrowband modes require samplerate=8000")

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.cfg.bind_host, self.cfg.listen_port))
        self.sock.settimeout(0.2)

        self.seq = 0
        self.running = False
        self._sender_vad_state = 0
        self._play_q: queue.Queue[bytes] = queue.Queue(maxsize=max(2, self.cfg.recv_queue_frames))
        self._threads: List[threading.Thread] = []
        self._input_stream = None
        self._output_stream = None
        self.stats = AudioStats()
        self._last_ping_tx = 0.0
        self._last_peer_seen = 0.0
        self._lock = threading.Lock()

    def _next_seq(self) -> int:
        with self._lock:
            seq = self.seq
            self.seq = (self.seq + 1) & 0xFFFFFFFF
        return seq

    def _get_rx_codec(self, mode_id: int) -> Codec2:
        if mode_id not in self.rx_codecs:
            name = MODE_ID_TO_NAME.get(mode_id)
            if name is None:
                raise ValueError(f"Unsupported remote mode id: {mode_id}")
            self.rx_codecs[mode_id] = Codec2(name)
        return self.rx_codecs[mode_id]

    def start(self) -> None:
        if self.running:
            return
        self.running = True

        self._input_stream = sd.RawInputStream(
            samplerate=self.cfg.samplerate,
            channels=1,
            dtype="int16",
            blocksize=self.frame_samples,
            device=self.cfg.input_device,
            callback=self._on_input,
            latency="low",
        )
        self._output_stream = sd.RawOutputStream(
            samplerate=self.cfg.samplerate,
            channels=1,
            dtype="int16",
            blocksize=self.frame_samples,
            device=self.cfg.output_device,
            callback=self._on_output,
            latency="low",
        )

        self._input_stream.start()
        self._output_stream.start()

        for target in [self._recv_loop, self._ping_loop]:
            t = threading.Thread(target=target, daemon=True)
            t.start()
            self._threads.append(t)

        bitrate = self.tx_codec.bits_per_frame * self.cfg.samplerate / self.frame_samples
        frame_ms = 1000.0 * self.frame_samples / self.cfg.samplerate
        self.log(
            f"Started: listen={self.cfg.listen_port} peer={self.cfg.peer_host}:{self.cfg.peer_port} "
            f"codec2={self.cfg.codec_mode} frame={self.frame_samples} samples ({frame_ms:.1f} ms) "
            f"raw_codec_rate≈{bitrate:.0f} bps input={describe_device(self.cfg.input_device)} "
            f"output={describe_device(self.cfg.output_device)}"
        )

    def stop(self) -> None:
        if not self.running:
            return
        self.running = False
        time.sleep(0.05)
        try:
            if self._input_stream:
                self._input_stream.stop()
                self._input_stream.close()
        finally:
            self._input_stream = None
        try:
            if self._output_stream:
                self._output_stream.stop()
                self._output_stream.close()
        finally:
            self._output_stream = None
        try:
            self.sock.close()
        except Exception:
            pass
        self.tx_codec.close()
        for c in self.rx_codecs.values():
            c.close()
        self.rx_codecs.clear()
        self.log("Stopped")

    def _sendto(self, payload: bytes) -> None:
        self.sock.sendto(payload, (self.cfg.peer_host, self.cfg.peer_port))
        self.stats.tx_packets += 1
        self.stats.tx_bytes_udp_payload += len(payload)

    def send_ping(self) -> None:
        pkt = build_packet(PKT_PING, self.tx_codec.mode_id, 0, self._next_seq(), b"")
        self._sendto(pkt)

    def send_test(self) -> None:
        token = struct.pack("!Q", time.monotonic_ns())
        pkt = build_packet(PKT_TEST, self.tx_codec.mode_id, 0, self._next_seq(), token)
        self._sendto(pkt)
        self.stats.test_requests += 1
        self.log("Connection test packet sent")

    def _on_input(self, indata, frames, time_info, status) -> None:
        if status:
            self.log(f"Input status: {status}")
        pcm = np.frombuffer(indata, dtype=np.int16).copy()
        if self.cfg.input_gain != 1.0:
            pcm_f = np.clip(pcm.astype(np.float32) * self.cfg.input_gain, -32768, 32767)
            pcm = pcm_f.astype(np.int16)
        if self.cfg.preemphasis > 0.0:
            x = pcm.astype(np.float32)
            y = np.empty_like(x)
            y[0] = x[0]
            y[1:] = x[1:] - self.cfg.preemphasis * x[:-1]
            pcm = np.clip(y, -32768, 32767).astype(np.int16)

        rms = int(np.sqrt(np.mean(pcm.astype(np.float32) ** 2) + 1e-9))
        voice = rms >= self.cfg.vad_threshold
        if voice:
            self._sender_vad_state = self.cfg.vad_hangover_frames
        elif self._sender_vad_state > 0:
            self._sender_vad_state -= 1
            voice = True

        if not voice:
            self.stats.tx_dtx_drops += 1
            return

        encoded = self.tx_codec.encode(pcm)
        packet = build_packet(PKT_AUDIO, self.tx_codec.mode_id, 0, self._next_seq(), encoded)
        try:
            self._sendto(packet)
            self.stats.tx_voice_frames += 1
        except OSError as e:
            self.log(f"UDP send error: {e}")

    def _on_output(self, outdata, frames, time_info, status) -> None:
        if status:
            self.log(f"Output status: {status}")
        try:
            frame = self._play_q.get_nowait()
        except queue.Empty:
            frame = bytes(self.frame_samples * 2)
            self.stats.rx_silence_fill_frames += 1
        outdata[:] = frame

    def _recv_loop(self) -> None:
        while self.running:
            try:
                data, _addr = self.sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break
            parsed = parse_packet(data)
            if not parsed:
                continue
            pkt_type, mode_id, flags, seq, payload = parsed
            self._last_peer_seen = time.time()
            self.stats.rx_packets += 1
            self.stats.rx_bytes_udp_payload += len(data)

            if pkt_type == PKT_PING:
                try:
                    pkt = build_packet(PKT_PONG, self.tx_codec.mode_id, 0, self._next_seq(), b"")
                    self._sendto(pkt)
                except OSError:
                    pass
                continue

            if pkt_type == PKT_PONG:
                self.stats.last_pong_at = time.time()
                continue

            if pkt_type == PKT_TEST:
                try:
                    pkt = build_packet(PKT_TEST_REPLY, self.tx_codec.mode_id, 0, self._next_seq(), payload)
                    self._sendto(pkt)
                except OSError:
                    pass
                continue

            if pkt_type == PKT_TEST_REPLY:
                self.stats.test_replies += 1
                self.stats.last_test_reply_at = time.time()
                if len(payload) == 8:
                    sent_ns = struct.unpack("!Q", payload)[0]
                    self.stats.last_test_rtt_ms = (time.monotonic_ns() - sent_ns) / 1e6
                    self.log(f"Connection test reply received: RTT={self.stats.last_test_rtt_ms:.1f} ms")
                else:
                    self.log("Connection test reply received")
                continue

            if self.stats.last_rx_seq is not None and seq != ((self.stats.last_rx_seq + 1) & 0xFFFFFFFF):
                gap = (seq - self.stats.last_rx_seq - 1) & 0xFFFFFFFF
                if gap < (1 << 31):
                    self.stats.rx_seq_gaps += gap
            self.stats.last_rx_seq = seq

            if pkt_type != PKT_AUDIO:
                continue
            try:
                decoded = self._get_rx_codec(mode_id).decode(payload)
            except Exception as e:
                self.log(f"Decode error: {e}")
                continue

            frame_bytes = decoded.astype(np.int16, copy=False).tobytes()
            try:
                self._play_q.put_nowait(frame_bytes)
            except queue.Full:
                try:
                    _ = self._play_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._play_q.put_nowait(frame_bytes)
                except queue.Full:
                    pass

    def _ping_loop(self) -> None:
        while self.running:
            now = time.time()
            if now - self._last_ping_tx >= self.cfg.ping_interval_s:
                self._last_ping_tx = now
                try:
                    self.send_ping()
                except OSError:
                    pass
            time.sleep(0.05)

    def connection_state(self) -> str:
        now = time.time()
        peer_age = now - self._last_peer_seen if self._last_peer_seen else float("inf")
        pong_age = now - self.stats.last_pong_at if self.stats.last_pong_at else float("inf")
        test_age = now - self.stats.last_test_reply_at if self.stats.last_test_reply_at else float("inf")

        if peer_age < self.cfg.ttl_ping_s and pong_age < self.cfg.ttl_ping_s * 2.0:
            base = "connected"
        elif peer_age < self.cfg.ttl_ping_s * 2.0:
            base = "rx-only"
        else:
            base = "waiting"

        parts = [base]
        if peer_age < 999:
            parts.append(f"last_rx={peer_age:.1f}s")
        if pong_age < 999:
            parts.append(f"last_pong={pong_age:.1f}s")
        if self.stats.last_test_rtt_ms is not None and test_age < 30.0:
            parts.append(f"rtt={self.stats.last_test_rtt_ms:.1f}ms")
        return " ".join(parts)

    def status_text(self) -> str:
        frame_ms = 1000.0 * self.frame_samples / self.cfg.samplerate
        return (
            f"codec2={self.cfg.codec_mode} frame={frame_ms:.1f}ms avg_tx={self.stats.avg_kbps():.2f}kbps "
            f"tx_voice={self.stats.tx_voice_frames} dtx_drop={self.stats.tx_dtx_drops} "
            f"rx_gap={self.stats.rx_seq_gaps} link={self.connection_state()} "
            f"in={describe_device(self.cfg.input_device)} out={describe_device(self.cfg.output_device)}"
        )


# =========================
# GUI
# =========================

def run_gui(default_args: argparse.Namespace) -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext, ttk

    root = tk.Tk()
    root.title("Codec2 Audio Cross Bridge")
    root.geometry("1080x820")

    bridge_ref = {"bridge": None}
    current_config_path = {"path": default_args.config or ""}

    vars_map = {
        "peer_host": tk.StringVar(value=default_args.peer_host or "127.0.0.1"),
        "peer_port": tk.StringVar(value=str(default_args.peer_port or 5001)),
        "listen_port": tk.StringVar(value=str(default_args.listen_port or 5000)),
        "codec_mode": tk.StringVar(value=default_args.codec_mode),
        "vad_threshold": tk.StringVar(value=str(default_args.vad_threshold)),
        "vad_hangover_frames": tk.StringVar(value=str(default_args.vad_hangover_frames)),
        "input_gain": tk.StringVar(value=str(default_args.input_gain)),
        "preemphasis": tk.StringVar(value=str(default_args.preemphasis)),
        "recv_queue_frames": tk.StringVar(value=str(default_args.recv_queue_frames)),
        "input_device": tk.StringVar(value="" if default_args.input_device is None else str(default_args.input_device)),
        "output_device": tk.StringVar(value="" if default_args.output_device is None else str(default_args.output_device)),
        "input_device_name": tk.StringVar(value=default_args.input_device_name or ""),
        "output_device_name": tk.StringVar(value=default_args.output_device_name or ""),
    }

    def log(msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        text.configure(state="normal")
        text.insert("end", f"[{ts}] {msg}\n")
        text.see("end")
        text.configure(state="disabled")

    def refresh_devices() -> None:
        try:
            devs = query_audio_devices()
        except Exception as e:
            log(f"Device query failed: {e}")
            return

        input_values = [""]
        output_values = [""]
        for i, dev in enumerate(devs):
            label = f"[{i}] {dev['name']}"
            if int(dev.get("max_input_channels", 0)) > 0:
                input_values.append(label)
            if int(dev.get("max_output_channels", 0)) > 0:
                output_values.append(label)
        input_combo.configure(values=input_values)
        output_combo.configure(values=output_values)
        log("Device list refreshed")

    def sync_name_to_id(direction: str) -> None:
        name_key = f"{direction}_device_name"
        id_key = f"{direction}_device"
        raw = vars_map[name_key].get().strip()
        if raw.startswith("[") and "]" in raw:
            maybe_id = raw[1:raw.index("]")].strip()
            if maybe_id.isdigit():
                vars_map[id_key].set(maybe_id)
                vars_map[name_key].set(raw[raw.index("]") + 1:].strip())

    def int_or_none(s: str) -> Optional[int]:
        s = s.strip()
        return None if s == "" else int(s)

    def build_config() -> AudioConfig:
        return AudioConfig(
            peer_host=vars_map["peer_host"].get().strip(),
            peer_port=int(vars_map["peer_port"].get()),
            listen_port=int(vars_map["listen_port"].get()),
            codec_mode=vars_map["codec_mode"].get().strip(),
            samplerate=8000,
            input_device=int_or_none(vars_map["input_device"].get()),
            output_device=int_or_none(vars_map["output_device"].get()),
            input_device_name=vars_map["input_device_name"].get().strip() or None,
            output_device_name=vars_map["output_device_name"].get().strip() or None,
            vad_threshold=int(vars_map["vad_threshold"].get()),
            vad_hangover_frames=int(vars_map["vad_hangover_frames"].get()),
            preemphasis=float(vars_map["preemphasis"].get()),
            input_gain=float(vars_map["input_gain"].get()),
            recv_queue_frames=int(vars_map["recv_queue_frames"].get()),
        )

    def apply_config_to_form(data: Dict[str, Any]) -> None:
        mapping = {
            "peer_host": "peer_host",
            "peer_port": "peer_port",
            "listen_port": "listen_port",
            "codec_mode": "codec_mode",
            "vad_threshold": "vad_threshold",
            "vad_hangover_frames": "vad_hangover_frames",
            "input_gain": "input_gain",
            "preemphasis": "preemphasis",
            "recv_queue_frames": "recv_queue_frames",
            "input_device": "input_device",
            "output_device": "output_device",
            "input_device_name": "input_device_name",
            "output_device_name": "output_device_name",
        }
        for src, dst in mapping.items():
            if src in data and data[src] is not None:
                vars_map[dst].set(str(data[src]))
            elif src in {"input_device", "output_device", "input_device_name", "output_device_name"} and src in data:
                vars_map[dst].set("")

    def save_config_dialog() -> None:
        try:
            cfg = build_config()
        except Exception as e:
            messagebox.showerror("Save failed", str(e))
            return
        path = filedialog.asksaveasfilename(
            title="Save config YAML",
            defaultextension=".yaml",
            filetypes=[("YAML", "*.yaml *.yml"), ("All files", "*.*")],
            initialfile=os.path.basename(current_config_path["path"] or "audio_cross_codec2.yaml"),
        )
        if not path:
            return
        save_config_yaml(path, asdict(cfg))
        current_config_path["path"] = path
        config_path_var.set(path)
        log(f"Config saved: {path}")

    def load_config_dialog() -> None:
        path = filedialog.askopenfilename(
            title="Load config YAML",
            filetypes=[("YAML", "*.yaml *.yml"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            data = load_config_yaml(path)
            apply_config_to_form(data)
            current_config_path["path"] = path
            config_path_var.set(path)
            log(f"Config loaded: {path}")
        except Exception as e:
            messagebox.showerror("Load failed", str(e))
            log(f"Load failed: {e}")

    def start_bridge() -> None:
        if bridge_ref["bridge"] is not None:
            return
        try:
            cfg = build_config()
            bridge = AudioBridge(cfg, logger=log)
            bridge.start()
            bridge_ref["bridge"] = bridge
            start_btn.configure(state="disabled")
            stop_btn.configure(state="normal")
            test_btn.configure(state="normal")
        except Exception as e:
            messagebox.showerror("Start failed", str(e))
            log(f"Start failed: {e}")

    def stop_bridge() -> None:
        bridge = bridge_ref.get("bridge")
        if bridge is None:
            return
        try:
            bridge.stop()
        except Exception as e:
            log(f"Stop error: {e}")
        bridge_ref["bridge"] = None
        start_btn.configure(state="normal")
        stop_btn.configure(state="disabled")
        test_btn.configure(state="disabled")

    def test_connection() -> None:
        bridge = bridge_ref.get("bridge")
        if bridge is None:
            messagebox.showinfo("Not running", "Start the bridge first.")
            return
        try:
            bridge.send_test()
        except Exception as e:
            log(f"Connection test failed to send: {e}")

    def refresh_status() -> None:
        bridge = bridge_ref.get("bridge")
        if bridge is None:
            status_var.set("stopped")
            conn_var.set("link=stopped")
        else:
            status_var.set(bridge.status_text())
            conn_var.set(bridge.connection_state())
        root.after(300, refresh_status)

    def on_close() -> None:
        stop_bridge()
        root.destroy()

    main = ttk.Frame(root, padding=10)
    main.pack(fill="both", expand=True)

    form = ttk.Frame(main)
    form.pack(fill="x")

    basic_entries = [
        ("Peer host", "peer_host"),
        ("Peer port", "peer_port"),
        ("Listen port", "listen_port"),
        ("Codec2 mode", "codec_mode"),
        ("VAD threshold", "vad_threshold"),
        ("VAD hangover frames", "vad_hangover_frames"),
        ("Input gain", "input_gain"),
        ("Preemphasis", "preemphasis"),
        ("Recv queue frames", "recv_queue_frames"),
    ]

    row = 0
    for label, key in basic_entries:
        ttk.Label(form, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=4)
        if key == "codec_mode":
            ttk.Combobox(
                form,
                textvariable=vars_map[key],
                values=list(CODEC2_MODES.keys()),
                state="readonly",
            ).grid(row=row, column=1, sticky="ew", padx=4, pady=4)
        else:
            ttk.Entry(form, textvariable=vars_map[key]).grid(row=row, column=1, sticky="ew", padx=4, pady=4)
        row += 1

    ttk.Separator(form, orient="horizontal").grid(row=row, column=0, columnspan=4, sticky="ew", pady=(8, 8))
    row += 1

    ttk.Label(form, text="Input device name").grid(row=row, column=0, sticky="w", padx=4, pady=4)
    input_combo = ttk.Combobox(form, textvariable=vars_map["input_device_name"])
    input_combo.grid(row=row, column=1, sticky="ew", padx=4, pady=4)
    ttk.Button(
        form,
        text="Use default",
        command=lambda: (vars_map["input_device_name"].set(""), vars_map["input_device"].set("")),
    ).grid(row=row, column=2, sticky="ew", padx=4, pady=4)
    row += 1

    ttk.Label(form, text="Input device id").grid(row=row, column=0, sticky="w", padx=4, pady=4)
    ttk.Entry(form, textvariable=vars_map["input_device"]).grid(row=row, column=1, sticky="ew", padx=4, pady=4)
    ttk.Label(form, text="Name優先、空ならid、両方空なら既定").grid(row=row, column=2, columnspan=2, sticky="w", padx=4, pady=4)
    row += 1

    ttk.Label(form, text="Output device name").grid(row=row, column=0, sticky="w", padx=4, pady=4)
    output_combo = ttk.Combobox(form, textvariable=vars_map["output_device_name"])
    output_combo.grid(row=row, column=1, sticky="ew", padx=4, pady=4)
    ttk.Button(
        form,
        text="Use default",
        command=lambda: (vars_map["output_device_name"].set(""), vars_map["output_device"].set("")),
    ).grid(row=row, column=2, sticky="ew", padx=4, pady=4)
    row += 1

    ttk.Label(form, text="Output device id").grid(row=row, column=0, sticky="w", padx=4, pady=4)
    ttk.Entry(form, textvariable=vars_map["output_device"]).grid(row=row, column=1, sticky="ew", padx=4, pady=4)
    row += 1

    ttk.Separator(form, orient="horizontal").grid(row=row, column=0, columnspan=4, sticky="ew", pady=(8, 8))
    row += 1

    config_path_var = tk.StringVar(value=current_config_path["path"])
    conn_var = tk.StringVar(value="link=stopped")

    ttk.Label(form, text="Config file").grid(row=row, column=0, sticky="w", padx=4, pady=4)
    ttk.Entry(form, textvariable=config_path_var).grid(row=row, column=1, sticky="ew", padx=4, pady=4)
    ttk.Button(form, text="Load YAML", command=load_config_dialog).grid(row=row, column=2, sticky="ew", padx=4, pady=4)
    ttk.Button(form, text="Save YAML", command=save_config_dialog).grid(row=row, column=3, sticky="ew", padx=4, pady=4)
    row += 1

    ttk.Label(form, text="Connection").grid(row=row, column=0, sticky="w", padx=4, pady=4)
    ttk.Label(form, textvariable=conn_var).grid(row=row, column=1, columnspan=3, sticky="w", padx=4, pady=4)

    form.columnconfigure(1, weight=1)
    input_combo.bind("<<ComboboxSelected>>", lambda _e: sync_name_to_id("input"))
    output_combo.bind("<<ComboboxSelected>>", lambda _e: sync_name_to_id("output"))

    btns = ttk.Frame(main)
    btns.pack(fill="x", pady=(8, 6))
    start_btn = ttk.Button(btns, text="Start", command=start_bridge)
    stop_btn = ttk.Button(btns, text="Stop", command=stop_bridge, state="disabled")
    devices_btn = ttk.Button(btns, text="Show devices", command=lambda: log(list_audio_devices()))
    refresh_btn = ttk.Button(btns, text="Refresh devices", command=refresh_devices)
    test_btn = ttk.Button(btns, text="Test connection", command=test_connection, state="disabled")
    start_btn.pack(side="left", padx=4)
    stop_btn.pack(side="left", padx=4)
    test_btn.pack(side="left", padx=4)
    devices_btn.pack(side="left", padx=4)
    refresh_btn.pack(side="left", padx=4)

    status_var = tk.StringVar(value="stopped")
    ttk.Label(main, textvariable=status_var).pack(fill="x", pady=(0, 8))

    notes = ttk.Label(
        main,
        text=(
            "Mode guide: 3200=least degraded, 1200=most compressed. "
            "Input and output devices can be selected separately by name or by id. "
            "Name is matched first. Leave both empty to use the system default. "
            "Use Test connection after both sides are started. Saving/loading YAML stores ports, peer host, "
            "codec and device selection."
        ),
        wraplength=1020,
        justify="left",
    )
    notes.pack(fill="x", pady=(0, 8))

    text = scrolledtext.ScrolledText(main, height=22, state="disabled")
    text.pack(fill="both", expand=True)

    refresh_devices()
    root.protocol("WM_DELETE_WINDOW", on_close)
    refresh_status()
    root.mainloop()


# =========================
# CLI
# =========================

def build_default_arg_map() -> Dict[str, Any]:
    return {
        "peer_host": "127.0.0.1",
        "peer_port": 5001,
        "listen_port": 5000,
        "codec_mode": "1300",
        "samplerate": 8000,
        "input_device": None,
        "output_device": None,
        "input_device_name": None,
        "output_device_name": None,
        "vad_threshold": 700,
        "vad_hangover_frames": 2,
        "input_gain": 1.0,
        "preemphasis": 0.0,
        "recv_queue_frames": 4,
        "config": None,
        "save_config": None,
        "list_devices": False,
        "test_connection": False,
        "gui": False,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Codec2 bidirectional low-bandwidth audio bridge")
    p.add_argument("--gui", action="store_true", help="Launch Tk GUI")
    p.add_argument("--config", type=str, default=None, help="Load YAML config")
    p.add_argument("--save-config", type=str, default=None, help="Save current effective config YAML and continue")
    p.add_argument("--peer-host", type=str, default="127.0.0.1")
    p.add_argument("--peer-port", type=int, default=5001)
    p.add_argument("--listen-port", type=int, default=5000)
    p.add_argument("--codec-mode", choices=list(CODEC2_MODES.keys()), default="1300",
                   help="Lower mode => less bandwidth, more degradation")
    p.add_argument("--samplerate", type=int, default=8000, help="Codec2 narrowband requires 8000")
    p.add_argument("--input-device", type=int, default=None)
    p.add_argument("--output-device", type=int, default=None)
    p.add_argument("--input-device-name", type=str, default=None,
                   help="Exact or partial input device name. Used when --input-device is omitted")
    p.add_argument("--output-device-name", type=str, default=None,
                   help="Exact or partial output device name. Used when --output-device is omitted")
    p.add_argument("--vad-threshold", type=int, default=700,
                   help="RMS threshold. Higher => less traffic, more clipping risk")
    p.add_argument("--vad-hangover-frames", type=int, default=2)
    p.add_argument("--input-gain", type=float, default=1.0)
    p.add_argument("--preemphasis", type=float, default=0.0)
    p.add_argument("--recv-queue-frames", type=int, default=4)
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("--test-connection", action="store_true",
                   help="After start, send one test packet and print RTT when reply arrives")
    return p.parse_args()


def apply_config_to_args(args: argparse.Namespace) -> argparse.Namespace:
    if not args.config:
        return args
    cfg = load_config_yaml(args.config)
    defaults = build_default_arg_map()
    for key, value in cfg.items():
        attr = key
        if not hasattr(args, attr):
            continue
        if getattr(args, attr) == defaults.get(attr):
            setattr(args, attr, value)
    return args


def effective_config_from_args(args: argparse.Namespace) -> AudioConfig:
    return AudioConfig(
        peer_host=args.peer_host,
        peer_port=args.peer_port,
        listen_port=args.listen_port,
        codec_mode=args.codec_mode,
        samplerate=args.samplerate,
        input_device=args.input_device,
        output_device=args.output_device,
        input_device_name=args.input_device_name,
        output_device_name=args.output_device_name,
        vad_threshold=args.vad_threshold,
        vad_hangover_frames=args.vad_hangover_frames,
        input_gain=args.input_gain,
        preemphasis=args.preemphasis,
        recv_queue_frames=args.recv_queue_frames,
    )


def run_cli(args: argparse.Namespace) -> int:
    if args.list_devices:
        print(list_audio_devices())
        return 0

    cfg = effective_config_from_args(args)
    if args.save_config:
        save_config_yaml(args.save_config, asdict(cfg))
        print(f"Saved config: {args.save_config}", flush=True)

    bridge = AudioBridge(cfg)
    bridge.start()
    print("Press Ctrl+C to stop.", flush=True)
    if args.test_connection:
        time.sleep(0.3)
        bridge.send_test()
    try:
        while True:
            time.sleep(1.0)
            print(bridge.status_text(), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.stop()
    return 0


if __name__ == "__main__":
    args = parse_args()
    args = apply_config_to_args(args)
    if args.gui or len(sys.argv) == 1:
        run_gui(args)
    else:
        raise SystemExit(run_cli(args))
