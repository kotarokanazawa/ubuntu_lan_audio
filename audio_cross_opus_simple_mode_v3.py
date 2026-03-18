#!/usr/bin/env python3
"""
Discord-like low-bandwidth Opus audio bridge for Ubuntu/Linux.

Key points
- Full duplex UDP mic<->speaker bridge
- Opus voice compression
- Discord-like dark UI
- Live-adjustable gains / VAD / lightweight DSP
- "Complete AEC" mode using PulseAudio/PipeWire module-echo-cancel (WebRTC/Speex backends)
- YAML save/load
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import math
import os
import queue
import shlex
import socket
import struct
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import sounddevice as sd
except Exception as e:
    sd = None
    _SOUNDDEVICE_IMPORT_ERROR = e
else:
    _SOUNDDEVICE_IMPORT_ERROR = None

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


# =========================
# Opus
# =========================

_OPUS_LIB = ctypes.util.find_library("opus")
if not _OPUS_LIB:
    raise RuntimeError("libopus not found. Install: sudo apt install libopus0 libopus-dev")

_libopus = ctypes.CDLL(_OPUS_LIB)

OPUS_APPLICATIONS = {"voip": 2048, "audio": 2049, "restricted_lowdelay": 2051}
OPUS_BANDWIDTHS = {
    "auto": -1000,
    "narrowband": 1101,
    "mediumband": 1102,
    "wideband": 1103,
    "superwideband": 1104,
    "fullband": 1105,
}
OPUS_SIGNAL = {"auto": -1000, "voice": 3001, "music": 3002}

OPUS_SET_BITRATE_REQUEST = 4002
OPUS_SET_COMPLEXITY_REQUEST = 4010
OPUS_SET_INBAND_FEC_REQUEST = 4012
OPUS_SET_PACKET_LOSS_PERC_REQUEST = 4014
OPUS_SET_DTX_REQUEST = 4016
OPUS_SET_VBR_REQUEST = 4006
OPUS_SET_APPLICATION_REQUEST = 4000
OPUS_SET_BANDWIDTH_REQUEST = 4008
OPUS_SET_SIGNAL_REQUEST = 4024

_libopus.opus_encoder_create.argtypes = [ctypes.c_int32, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
_libopus.opus_encoder_create.restype = ctypes.c_void_p
_libopus.opus_encoder_destroy.argtypes = [ctypes.c_void_p]
_libopus.opus_encoder_destroy.restype = None
_libopus.opus_encode.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int16),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_ubyte),
    ctypes.c_int32,
]
_libopus.opus_encode.restype = ctypes.c_int32
_libopus.opus_encoder_ctl.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_libopus.opus_encoder_ctl.restype = ctypes.c_int
_libopus.opus_decoder_create.argtypes = [ctypes.c_int32, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
_libopus.opus_decoder_create.restype = ctypes.c_void_p
_libopus.opus_decoder_destroy.argtypes = [ctypes.c_void_p]
_libopus.opus_decoder_destroy.restype = None
_libopus.opus_decode.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_ubyte),
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int16),
    ctypes.c_int,
    ctypes.c_int,
]
_libopus.opus_decode.restype = ctypes.c_int
_libopus.opus_strerror.argtypes = [ctypes.c_int]
_libopus.opus_strerror.restype = ctypes.c_char_p


class OpusError(RuntimeError):
    pass


def opus_error_text(code: int) -> str:
    try:
        return _libopus.opus_strerror(int(code)).decode("utf-8", errors="replace")
    except Exception:
        return f"Opus error {code}"


class OpusCodec:
    def __init__(
        self,
        samplerate: int,
        channels: int,
        frame_ms: int,
        bitrate: int,
        complexity: int,
        application: str,
        bandwidth: str,
        signal_type: str,
        vbr: bool,
        inband_fec: bool,
        packet_loss_perc: int,
        dtx: bool,
    ):
        if samplerate not in (8000, 12000, 16000, 24000, 48000):
            raise ValueError("Opus samplerate must be one of 8000, 12000, 16000, 24000, 48000")
        if channels != 1:
            raise ValueError("mono only")
        if frame_ms not in (2, 5, 10, 20, 40, 60):
            raise ValueError("frame_ms must be 2/5/10/20/40/60")
        self.samplerate = samplerate
        self.channels = channels
        self.frame_ms = frame_ms
        self.frame_samples = int(self.samplerate * self.frame_ms / 1000)
        self.max_packet_bytes = 4000

        err = ctypes.c_int(0)
        app_id = OPUS_APPLICATIONS[application]
        self.enc = _libopus.opus_encoder_create(self.samplerate, self.channels, app_id, ctypes.byref(err))
        if err.value != 0 or not self.enc:
            raise OpusError(f"opus_encoder_create failed: {opus_error_text(err.value)}")
        self.dec = _libopus.opus_decoder_create(self.samplerate, self.channels, ctypes.byref(err))
        if err.value != 0 or not self.dec:
            if self.enc:
                _libopus.opus_encoder_destroy(self.enc)
            raise OpusError(f"opus_decoder_create failed: {opus_error_text(err.value)}")

        self._ctl(self.enc, OPUS_SET_BITRATE_REQUEST, int(bitrate))
        self._ctl(self.enc, OPUS_SET_COMPLEXITY_REQUEST, int(complexity))
        self._ctl(self.enc, OPUS_SET_VBR_REQUEST, 1 if vbr else 0)
        self._ctl(self.enc, OPUS_SET_INBAND_FEC_REQUEST, 1 if inband_fec else 0)
        self._ctl(self.enc, OPUS_SET_PACKET_LOSS_PERC_REQUEST, int(packet_loss_perc))
        self._ctl(self.enc, OPUS_SET_DTX_REQUEST, 1 if dtx else 0)
        self._ctl(self.enc, OPUS_SET_APPLICATION_REQUEST, app_id)
        self._ctl(self.enc, OPUS_SET_BANDWIDTH_REQUEST, OPUS_BANDWIDTHS[bandwidth])
        self._ctl(self.enc, OPUS_SET_SIGNAL_REQUEST, OPUS_SIGNAL[signal_type])

    @staticmethod
    def _ctl(handle, request: int, value: int) -> None:
        rc = _libopus.opus_encoder_ctl(handle, request, value)
        if rc != 0:
            raise OpusError(f"opus_encoder_ctl({request}, {value}) failed: {opus_error_text(rc)}")

    def encode(self, pcm_i16: np.ndarray) -> bytes:
        out = (ctypes.c_ubyte * self.max_packet_bytes)()
        inp = pcm_i16.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
        n = _libopus.opus_encode(self.enc, inp, self.frame_samples, out, self.max_packet_bytes)
        if n < 0:
            raise OpusError(f"opus_encode failed: {opus_error_text(n)}")
        return bytes(out[:n])

    def decode(self, payload: bytes) -> np.ndarray:
        out = np.zeros(self.frame_samples, dtype=np.int16)
        out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
        inbuf = (ctypes.c_ubyte * len(payload)).from_buffer_copy(payload) if payload else None
        n = _libopus.opus_decode(self.dec, inbuf, len(payload), out_ptr, self.frame_samples, 0)
        if n < 0:
            raise OpusError(f"opus_decode failed: {opus_error_text(n)}")
        if n < self.frame_samples:
            out[n:] = 0
        return out

    def close(self) -> None:
        if getattr(self, "enc", None):
            _libopus.opus_encoder_destroy(self.enc)
            self.enc = None
        if getattr(self, "dec", None):
            _libopus.opus_decoder_destroy(self.dec)
            self.dec = None


# =========================
# Protocol
# =========================

MAGIC = b"OPA1"
PROTO_VERSION = 1
PKT_AUDIO = 1
PKT_PING = 2
PKT_PONG = 3
PKT_TEST = 4
PKT_TEST_REPLY = 5
HEADER_STRUCT = struct.Struct("!4sBBBBIIH")


def build_packet(pkt_type: int, flags: int, seq: int, samplerate: int, frame_ms: int, payload: bytes) -> bytes:
    return HEADER_STRUCT.pack(MAGIC, PROTO_VERSION, pkt_type, flags, 0, seq, samplerate, frame_ms) + payload


def parse_packet(data: bytes):
    if len(data) < HEADER_STRUCT.size:
        return None
    magic, ver, pkt_type, flags, _reserved, seq, samplerate, frame_ms = HEADER_STRUCT.unpack_from(data)
    if magic != MAGIC or ver != PROTO_VERSION:
        return None
    return pkt_type, flags, seq, samplerate, frame_ms, data[HEADER_STRUCT.size:]


# =========================
# Pulse / device helpers
# =========================

def ensure_sounddevice() -> None:
    if sd is None:
        raise RuntimeError(
            "sounddevice import failed. Install: pip install sounddevice"
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
        lines.append(f"[{i}] {dev['name']} in={dev['max_input_channels']} out={dev['max_output_channels']}{tag_txt}")
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


def pactl_available() -> bool:
    try:
        subprocess.run(["pactl", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False


def pulse_list_short(kind: str) -> str:
    try:
        out = subprocess.check_output(["pactl", "list", "short", kind], text=True)
        return out
    except Exception:
        return ""


class PulseAECManager:
    def __init__(self):
        self.module_id: Optional[str] = None
        self.source_name: Optional[str] = None
        self.sink_name: Optional[str] = None
        self.prev_default_source: Optional[str] = None
        self.prev_default_sink: Optional[str] = None

    def _get_defaults(self) -> Tuple[Optional[str], Optional[str]]:
        try:
            out = subprocess.check_output(["pactl", "info"], text=True, stderr=subprocess.DEVNULL)
        except Exception:
            return None, None
        src = None
        sink = None
        for line in out.splitlines():
            if line.startswith("Default Source:"):
                src = line.split(":", 1)[1].strip()
            elif line.startswith("Default Sink:"):
                sink = line.split(":", 1)[1].strip()
        return src, sink

    def unload(self) -> None:
        if self.prev_default_source:
            try:
                subprocess.run(["pactl", "set-default-source", self.prev_default_source], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
        if self.prev_default_sink:
            try:
                subprocess.run(["pactl", "set-default-sink", self.prev_default_sink], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
        if self.module_id:
            try:
                subprocess.run(["pactl", "unload-module", self.module_id], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
        self.module_id = None
        self.source_name = None
        self.sink_name = None
        self.prev_default_source = None
        self.prev_default_sink = None

    def load(
        self,
        input_master_hint: str,
        output_master_hint: str,
        method: str = "webrtc",
    ) -> Tuple[str, str]:
        self.unload()
        if not pactl_available():
            raise RuntimeError("pactl not available. Complete AEC requires PulseAudio/PipeWire pactl.")
        self.prev_default_source, self.prev_default_sink = self._get_defaults()
        tag = f"chatgpt_aec_{int(time.time())}"
        source_name = f"{tag}_source"
        sink_name = f"{tag}_sink"

        args = [
            "pactl", "load-module", "module-echo-cancel",
            f"source_name={source_name}",
            f"sink_name={sink_name}",
            f"aec_method={method}",
            "use_master_format=1",
        ]
        if input_master_hint:
            args.append(f"source_master={input_master_hint}")
        if output_master_hint:
            args.append(f"sink_master={output_master_hint}")

        proc = subprocess.run(args, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                "Failed to load module-echo-cancel.\n"
                f"Command: {' '.join(shlex.quote(a) for a in args)}\n"
                f"stderr: {proc.stderr.strip()}"
            )
        self.module_id = proc.stdout.strip()
        self.source_name = source_name
        self.sink_name = sink_name
        # Make the AEC virtual devices the system defaults; PortAudio/sounddevice can then open defaults reliably.
        subprocess.run(["pactl", "set-default-source", source_name], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["pactl", "set-default-sink", sink_name], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Let PipeWire/Pulse publish the new defaults before stream creation.
        time.sleep(0.25)
        return source_name, sink_name


# =========================
# YAML
# =========================

CONFIG_KEYS = {
    "peer_host", "peer_port", "listen_port",
    "samplerate", "opus_bitrate", "opus_frame_ms", "opus_complexity",
    "opus_application", "opus_bandwidth", "opus_signal_type", "opus_vbr",
    "opus_inband_fec", "opus_packet_loss_perc", "opus_dtx",
    "input_device", "output_device", "input_device_name", "output_device_name",
    "vad_threshold", "vad_hangover_frames", "preemphasis", "input_gain", "output_gain",
    "recv_queue_frames", "bind_host", "ttl_ping_s", "ping_interval_s",
    "dsp_dc_block", "dsp_noise_gate_enable", "dsp_noise_gate_threshold", "dsp_noise_gate_slope",
    "dsp_lowpass_enable", "dsp_lowpass_alpha", "dsp_echo_reduce_enable", "dsp_echo_subtract",
    "dsp_echo_gate", "dsp_output_limiter_enable", "dsp_output_limit",
    "aec_mode", "aec_method"
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


def save_config_yaml(path: str, data: Dict[str, Any]) -> None:
    data = {k: data.get(k) for k in CONFIG_KEYS}
    if yaml is not None:
        Path(path).write_text(yaml.safe_dump(data, sort_keys=True, allow_unicode=True), encoding="utf-8")
    else:
        Path(path).write_text("\n".join(f"{k}: {data.get(k)}" for k in sorted(data.keys())) + "\n", encoding="utf-8")


# =========================
# Data classes
# =========================

@dataclass
class AudioConfig:
    peer_host: str
    peer_port: int
    listen_port: int
    samplerate: int = 16000
    opus_bitrate: int = 8000
    opus_frame_ms: int = 20
    opus_complexity: int = 2
    opus_application: str = "voip"
    opus_bandwidth: str = "wideband"
    opus_signal_type: str = "voice"
    opus_vbr: bool = True
    opus_inband_fec: bool = False
    opus_packet_loss_perc: int = 0
    opus_dtx: bool = False
    input_device: Optional[int] = None
    output_device: Optional[int] = None
    input_device_name: Optional[str] = None
    output_device_name: Optional[str] = None
    vad_threshold: int = 700
    vad_hangover_frames: int = 2
    preemphasis: float = 0.0
    input_gain: float = 1.0
    output_gain: float = 1.0
    recv_queue_frames: int = 3
    bind_host: str = "0.0.0.0"
    ttl_ping_s: float = 1.0
    ping_interval_s: float = 0.5
    dsp_dc_block: bool = True
    dsp_noise_gate_enable: bool = True
    dsp_noise_gate_threshold: int = 280
    dsp_noise_gate_slope: float = 1.0
    dsp_lowpass_enable: bool = True
    dsp_lowpass_alpha: float = 0.22
    dsp_echo_reduce_enable: bool = True
    dsp_echo_subtract: float = 0.35
    dsp_echo_gate: int = 220
    dsp_output_limiter_enable: bool = True
    dsp_output_limit: float = 0.92
    aec_mode: str = "off"      # off / complete
    aec_method: str = "webrtc" # webrtc / speex


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
        self.cfg = config
        self.log = logger or (lambda msg: print(msg, flush=True))
        self.codec = OpusCodec(
            samplerate=config.samplerate, channels=1, frame_ms=config.opus_frame_ms,
            bitrate=config.opus_bitrate, complexity=config.opus_complexity,
            application=config.opus_application, bandwidth=config.opus_bandwidth,
            signal_type=config.opus_signal_type, vbr=config.opus_vbr,
            inband_fec=config.opus_inband_fec, packet_loss_perc=config.opus_packet_loss_perc,
            dtx=config.opus_dtx,
        )
        self.frame_samples = self.codec.frame_samples

        self.aec_manager = PulseAECManager()
        self.input_device_id = config.input_device
        self.output_device_id = config.output_device
        self._select_devices()

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
        self._dc_prev_x = 0.0
        self._dc_prev_y = 0.0
        self._lpf_prev_y = 0.0
        self._last_played = np.zeros(self.frame_samples, dtype=np.float32)

        self._vis_lock = threading.Lock()
        self._vis_len = max(self.cfg.samplerate // 2, self.frame_samples * 16)
        self._in_vis = np.zeros(self._vis_len, dtype=np.float32)
        self._rx_vis = np.zeros(self._vis_len, dtype=np.float32)
        self._in_rms = 0.0
        self._rx_rms = 0.0

    def _select_devices(self) -> None:
        # resolve raw selections first
        raw_in = resolve_device("input", self.cfg.input_device, self.cfg.input_device_name)
        raw_out = resolve_device("output", self.cfg.output_device, self.cfg.output_device_name)
        self.input_device_id = raw_in
        self.output_device_id = raw_out

        if self.cfg.aec_mode != "complete":
            return

        if not pactl_available():
            raise RuntimeError("Complete AEC requires pactl (PulseAudio/PipeWire).")

        in_name = describe_device(raw_in)
        out_name = describe_device(raw_out)
        src_name, sink_name = self.aec_manager.load(input_master_hint="", output_master_hint="", method=self.cfg.aec_method)
        # Use system defaults after module-echo-cancel is installed. This is more reliable than matching
        # the virtual device names back through PortAudio, which often exposes different display names.
        self.input_device_id = None
        self.output_device_id = None
        self.log(f"Complete AEC enabled: source={src_name} sink={sink_name} (base in={in_name}, out={out_name})")

    def apply_live_settings(self, cfg: AudioConfig) -> None:
        self.cfg.input_gain = cfg.input_gain
        self.cfg.output_gain = cfg.output_gain
        self.cfg.vad_threshold = cfg.vad_threshold
        self.cfg.vad_hangover_frames = cfg.vad_hangover_frames
        self.cfg.preemphasis = cfg.preemphasis
        self.cfg.dsp_dc_block = cfg.dsp_dc_block
        self.cfg.dsp_noise_gate_enable = cfg.dsp_noise_gate_enable
        self.cfg.dsp_noise_gate_threshold = cfg.dsp_noise_gate_threshold
        self.cfg.dsp_noise_gate_slope = cfg.dsp_noise_gate_slope
        self.cfg.dsp_lowpass_enable = cfg.dsp_lowpass_enable
        self.cfg.dsp_lowpass_alpha = cfg.dsp_lowpass_alpha
        self.cfg.dsp_echo_reduce_enable = cfg.dsp_echo_reduce_enable
        self.cfg.dsp_echo_subtract = cfg.dsp_echo_subtract
        self.cfg.dsp_echo_gate = cfg.dsp_echo_gate
        self.cfg.dsp_output_limiter_enable = cfg.dsp_output_limiter_enable
        self.cfg.dsp_output_limit = cfg.dsp_output_limit

    def _push_vis(self, which: str, pcm: np.ndarray) -> None:
        arr = pcm.astype(np.float32) / 32768.0
        n = min(len(arr), self._vis_len)
        with self._vis_lock:
            if which == "in":
                self._in_vis = np.roll(self._in_vis, -n)
                self._in_vis[-n:] = arr[-n:]
                self._in_rms = float(np.sqrt(np.mean(arr * arr) + 1e-12))
            else:
                self._rx_vis = np.roll(self._rx_vis, -n)
                self._rx_vis[-n:] = arr[-n:]
                self._rx_rms = float(np.sqrt(np.mean(arr * arr) + 1e-12))

    def get_visual_state(self) -> Dict[str, Any]:
        with self._vis_lock:
            return {
                "input_wave": self._in_vis.copy(),
                "rx_wave": self._rx_vis.copy(),
                "input_rms": self._in_rms,
                "rx_rms": self._rx_rms,
                "samplerate": self.cfg.samplerate,
            }

    def _next_seq(self) -> int:
        with self._lock:
            seq = self.seq
            self.seq = (self.seq + 1) & 0xFFFFFFFF
        return seq

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self._input_stream = sd.RawInputStream(
            samplerate=self.cfg.samplerate, channels=1, dtype="int16", blocksize=self.frame_samples,
            device=self.input_device_id, callback=self._on_input, latency="low",
        )
        self._output_stream = sd.RawOutputStream(
            samplerate=self.cfg.samplerate, channels=1, dtype="int16", blocksize=self.frame_samples,
            device=self.output_device_id, callback=self._on_output, latency="low",
        )
        self._input_stream.start()
        self._output_stream.start()
        for target in [self._recv_loop, self._ping_loop]:
            t = threading.Thread(target=target, daemon=True)
            t.start()
            self._threads.append(t)
        self.log(
            f"Started: listen={self.cfg.listen_port} peer={self.cfg.peer_host}:{self.cfg.peer_port} "
            f"opus={self.cfg.opus_bitrate}bps/{self.cfg.opus_frame_ms}ms "
            f"AEC={self.cfg.aec_mode}/{self.cfg.aec_method} "
            f"in={describe_device(self.input_device_id)} out={describe_device(self.output_device_id)}"
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
        self.codec.close()
        self.aec_manager.unload()
        self.log("Stopped")

    def _sendto(self, payload: bytes) -> None:
        self.sock.sendto(payload, (self.cfg.peer_host, self.cfg.peer_port))
        self.stats.tx_packets += 1
        self.stats.tx_bytes_udp_payload += len(payload)

    def send_ping(self) -> None:
        self._sendto(build_packet(PKT_PING, 0, self._next_seq(), self.cfg.samplerate, self.cfg.opus_frame_ms, b""))

    def send_test(self) -> None:
        token = struct.pack("!Q", time.monotonic_ns())
        self._sendto(build_packet(PKT_TEST, 0, self._next_seq(), self.cfg.samplerate, self.cfg.opus_frame_ms, token))
        self.stats.test_requests += 1
        self.log("Connection test packet sent")

    def _dc_block(self, x: np.ndarray) -> np.ndarray:
        if not self.cfg.dsp_dc_block:
            return x
        y = np.empty_like(x)
        prev_x = self._dc_prev_x
        prev_y = self._dc_prev_y
        r = 0.995
        for i, s in enumerate(x):
            out = s - prev_x + r * prev_y
            y[i] = out
            prev_x = s
            prev_y = out
        self._dc_prev_x = float(prev_x)
        self._dc_prev_y = float(prev_y)
        return y

    def _lowpass(self, x: np.ndarray) -> np.ndarray:
        if not self.cfg.dsp_lowpass_enable:
            return x
        alpha = float(np.clip(self.cfg.dsp_lowpass_alpha, 0.01, 1.0))
        y = np.empty_like(x)
        prev = self._lpf_prev_y
        for i, s in enumerate(x):
            prev = alpha * s + (1.0 - alpha) * prev
            y[i] = prev
        self._lpf_prev_y = float(prev)
        return y

    def _noise_gate(self, x: np.ndarray) -> np.ndarray:
        if not self.cfg.dsp_noise_gate_enable:
            return x
        rms = float(np.sqrt(np.mean(x * x) + 1e-12))
        th = float(self.cfg.dsp_noise_gate_threshold)
        if rms < th:
            slope = float(np.clip(self.cfg.dsp_noise_gate_slope, 0.0, 2.0))
            gain = slope * (rms / max(th, 1.0))
            return x * gain
        return x

    def _echo_reduce(self, x: np.ndarray) -> np.ndarray:
        if self.cfg.aec_mode == "complete":
            return x
        if not self.cfg.dsp_echo_reduce_enable or self._last_played.shape[0] != x.shape[0]:
            return x
        played_rms = float(np.sqrt(np.mean(self._last_played * self._last_played) + 1e-12))
        if played_rms < float(self.cfg.dsp_echo_gate):
            return x
        coeff = float(np.clip(self.cfg.dsp_echo_subtract, 0.0, 1.0))
        return x - coeff * self._last_played

    def _limiter(self, x: np.ndarray, limit: float) -> np.ndarray:
        lim = float(np.clip(limit, 0.1, 1.0)) * 32767.0
        return np.clip(x, -lim, lim)

    def _on_input(self, indata, frames, time_info, status) -> None:
        if status:
            self.log(f"Input status: {status}")
        x = np.frombuffer(indata, dtype=np.int16).astype(np.float32)
        if self.cfg.input_gain != 1.0:
            x *= float(self.cfg.input_gain)
        if self.cfg.preemphasis > 0.0:
            p = float(self.cfg.preemphasis)
            y = np.empty_like(x)
            y[0] = x[0]
            y[1:] = x[1:] - p * x[:-1]
            x = y
        x = self._dc_block(x)
        x = self._echo_reduce(x)
        x = self._lowpass(x)
        x = self._noise_gate(x)
        x = self._limiter(x, 0.98)
        pcm = np.clip(x, -32768, 32767).astype(np.int16)
        self._push_vis("in", pcm)
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
        try:
            encoded = self.codec.encode(pcm)
        except Exception as e:
            self.log(f"Encode error: {e}")
            return
        try:
            self._sendto(build_packet(PKT_AUDIO, 0, self._next_seq(), self.cfg.samplerate, self.cfg.opus_frame_ms, encoded))
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
        if self.cfg.output_gain != 1.0:
            y = np.frombuffer(frame, dtype=np.int16).astype(np.float32) * float(self.cfg.output_gain)
            if self.cfg.dsp_output_limiter_enable:
                y = self._limiter(y, self.cfg.dsp_output_limit)
            frame = y.astype(np.int16).tobytes()
        outdata[:] = frame
        self._last_played = np.frombuffer(frame, dtype=np.int16).astype(np.float32)

    def _recv_loop(self) -> None:
        while self.running:
            try:
                data, _addr = self.sock.recvfrom(8192)
            except socket.timeout:
                continue
            except OSError:
                break
            parsed = parse_packet(data)
            if not parsed:
                continue
            pkt_type, flags, seq, samplerate, frame_ms, payload = parsed
            self._last_peer_seen = time.time()
            self.stats.rx_packets += 1
            self.stats.rx_bytes_udp_payload += len(data)
            if pkt_type == PKT_PING:
                try:
                    self._sendto(build_packet(PKT_PONG, 0, self._next_seq(), self.cfg.samplerate, self.cfg.opus_frame_ms, b""))
                except OSError:
                    pass
                continue
            if pkt_type == PKT_PONG:
                self.stats.last_pong_at = time.time()
                continue
            if pkt_type == PKT_TEST:
                try:
                    self._sendto(build_packet(PKT_TEST_REPLY, 0, self._next_seq(), self.cfg.samplerate, self.cfg.opus_frame_ms, payload))
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
                continue
            if self.stats.last_rx_seq is not None and seq != ((self.stats.last_rx_seq + 1) & 0xFFFFFFFF):
                gap = (seq - self.stats.last_rx_seq - 1) & 0xFFFFFFFF
                if gap < (1 << 31):
                    self.stats.rx_seq_gaps += gap
            self.stats.last_rx_seq = seq
            if pkt_type != PKT_AUDIO:
                continue
            if samplerate != self.cfg.samplerate or frame_ms != self.cfg.opus_frame_ms:
                self.log(f"Remote format mismatch ignored: sr={samplerate} frame_ms={frame_ms}")
                continue
            try:
                decoded = self.codec.decode(payload)
            except Exception as e:
                self.log(f"Decode error: {e}")
                continue
            if self.cfg.dsp_output_limiter_enable:
                decoded = self._limiter(decoded.astype(np.float32), self.cfg.dsp_output_limit).astype(np.int16)
            self._push_vis("rx", decoded)
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
        if self.stats.last_test_rtt_ms is not None and (now - self.stats.last_test_reply_at) < 30.0:
            parts.append(f"rtt={self.stats.last_test_rtt_ms:.1f}ms")
        return " ".join(parts)

    def status_text(self) -> str:
        return (
            f"opus={self.cfg.opus_bitrate}bps frame={self.cfg.opus_frame_ms}ms sr={self.cfg.samplerate} "
            f"avg_tx={self.stats.avg_kbps():.2f}kbps tx_voice={self.stats.tx_voice_frames} "
            f"dtx_drop={self.stats.tx_dtx_drops} rx_gap={self.stats.rx_seq_gaps} "
            f"aec={self.cfg.aec_mode}/{self.cfg.aec_method} link={self.connection_state()}"
        )


# =========================
# CLI
# =========================

def effective_config_from_args(args: argparse.Namespace) -> AudioConfig:
    return AudioConfig(
        peer_host=args.peer_host, peer_port=args.peer_port, listen_port=args.listen_port,
        samplerate=args.samplerate, opus_bitrate=args.opus_bitrate, opus_frame_ms=args.opus_frame_ms,
        opus_complexity=args.opus_complexity, opus_application=args.opus_application,
        opus_bandwidth=args.opus_bandwidth, opus_signal_type=args.opus_signal_type,
        opus_vbr=args.opus_vbr, opus_inband_fec=args.opus_inband_fec,
        opus_packet_loss_perc=args.opus_packet_loss_perc, opus_dtx=args.opus_dtx,
        input_device=args.input_device, output_device=args.output_device,
        input_device_name=args.input_device_name, output_device_name=args.output_device_name,
        vad_threshold=args.vad_threshold, vad_hangover_frames=args.vad_hangover_frames,
        input_gain=args.input_gain, output_gain=args.output_gain,
        preemphasis=args.preemphasis, recv_queue_frames=args.recv_queue_frames,
        dsp_dc_block=not args.dsp_no_dc_block, dsp_noise_gate_enable=not args.dsp_noise_gate_disable,
        dsp_noise_gate_threshold=args.dsp_noise_gate_threshold, dsp_noise_gate_slope=args.dsp_noise_gate_slope,
        dsp_lowpass_enable=not args.dsp_lowpass_disable, dsp_lowpass_alpha=args.dsp_lowpass_alpha,
        dsp_echo_reduce_enable=not args.dsp_echo_reduce_disable, dsp_echo_subtract=args.dsp_echo_subtract,
        dsp_echo_gate=args.dsp_echo_gate, dsp_output_limiter_enable=not args.dsp_output_limiter_disable,
        dsp_output_limit=args.dsp_output_limit, aec_mode="complete", aec_method=args.aec_method,
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


def build_default_arg_map() -> Dict[str, Any]:
    return {
        "peer_host": "127.0.0.1",
        "peer_port": 5001,
        "listen_port": 5000,
        "samplerate": 16000,
        "opus_bitrate": 8000,
        "opus_frame_ms": 20,
        "opus_complexity": 2,
        "opus_application": "voip",
        "opus_bandwidth": "wideband",
        "opus_signal_type": "voice",
        "opus_vbr": False,
        "opus_inband_fec": False,
        "opus_packet_loss_perc": 0,
        "opus_dtx": False,
        "input_device": None,
        "output_device": None,
        "input_device_name": None,
        "output_device_name": None,
        "vad_threshold": 700,
        "vad_hangover_frames": 2,
        "preemphasis": 0.0,
        "input_gain": 1.0,
        "output_gain": 1.0,
        "recv_queue_frames": 3,
        "dsp_no_dc_block": False,
        "dsp_noise_gate_disable": False,
        "dsp_noise_gate_threshold": 280,
        "dsp_noise_gate_slope": 1.0,
        "dsp_lowpass_disable": False,
        "dsp_lowpass_alpha": 0.22,
        "dsp_echo_reduce_disable": False,
        "dsp_echo_subtract": 0.35,
        "dsp_echo_gate": 220,
        "dsp_output_limiter_disable": False,
        "dsp_output_limit": 0.92,
        "aec_mode": "complete",
        "aec_method": "webrtc",
        "config": None,
        "save_config": None,
        "list_devices": False,
        "test_connection": False,
        "gui": False,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Opus audio bridge with complete AEC option")
    p.add_argument("--gui", action="store_true")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--save-config", type=str, default=None)
    p.add_argument("--peer-host", type=str, default="127.0.0.1")
    p.add_argument("--peer-port", type=int, default=5001)
    p.add_argument("--listen-port", type=int, default=5000)
    p.add_argument("--samplerate", type=int, default=16000, choices=[8000, 12000, 16000, 24000, 48000])
    p.add_argument("--opus-bitrate", type=int, default=8000)
    p.add_argument("--opus-frame-ms", type=int, default=20, choices=[2, 5, 10, 20, 40, 60])
    p.add_argument("--opus-complexity", type=int, default=2)
    p.add_argument("--opus-application", choices=list(OPUS_APPLICATIONS.keys()), default="voip")
    p.add_argument("--opus-bandwidth", choices=list(OPUS_BANDWIDTHS.keys()), default="wideband")
    p.add_argument("--opus-signal-type", choices=list(OPUS_SIGNAL.keys()), default="voice")
    p.add_argument("--opus-vbr", action="store_true")
    p.add_argument("--opus-inband-fec", action="store_true")
    p.add_argument("--opus-packet-loss-perc", type=int, default=0)
    p.add_argument("--opus-dtx", action="store_true")
    p.add_argument("--input-device", type=int, default=None)
    p.add_argument("--output-device", type=int, default=None)
    p.add_argument("--input-device-name", type=str, default=None)
    p.add_argument("--output-device-name", type=str, default=None)
    p.add_argument("--vad-threshold", type=int, default=700)
    p.add_argument("--vad-hangover-frames", type=int, default=2)
    p.add_argument("--preemphasis", type=float, default=0.0)
    p.add_argument("--input-gain", type=float, default=1.0)
    p.add_argument("--output-gain", type=float, default=1.0)
    p.add_argument("--recv-queue-frames", type=int, default=3)
    p.add_argument("--dsp-no-dc-block", action="store_true")
    p.add_argument("--dsp-noise-gate-disable", action="store_true")
    p.add_argument("--dsp-noise-gate-threshold", type=int, default=280)
    p.add_argument("--dsp-noise-gate-slope", type=float, default=1.0)
    p.add_argument("--dsp-lowpass-disable", action="store_true")
    p.add_argument("--dsp-lowpass-alpha", type=float, default=0.22)
    p.add_argument("--dsp-echo-reduce-disable", action="store_true")
    p.add_argument("--dsp-echo-subtract", type=float, default=0.35)
    p.add_argument("--dsp-echo-gate", type=int, default=220)
    p.add_argument("--dsp-output-limiter-disable", action="store_true")
    p.add_argument("--dsp-output-limit", type=float, default=0.92)
    p.add_argument("--aec-mode", choices=["off", "complete"], default="off")
    p.add_argument("--aec-method", choices=["webrtc", "speex"], default="webrtc")
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("--test-connection", action="store_true")
    return p.parse_args()


def apply_config_to_args(args: argparse.Namespace) -> argparse.Namespace:
    if not args.config:
        return args
    cfg = load_config_yaml(args.config)
    defaults = build_default_arg_map()
    for key, value in cfg.items():
        if not hasattr(args, key):
            continue
        if getattr(args, key) == defaults.get(key):
            setattr(args, key, value)
    return args


# =========================
# GUI
# =========================


def run_gui(default_args: argparse.Namespace) -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext, ttk

    class MeterBar(tk.Canvas):
        def __init__(self, master, title: str, width=220, height=30, **kwargs):
            super().__init__(master, width=width, height=height, bg="#111111", highlightthickness=1,
                             highlightbackground="#2f2f2f", **kwargs)
            self.title = title

        def _color_for_norm(self, norm: float) -> str:
            norm = max(0.0, min(1.0, norm))
            if norm < 0.6:
                t = norm / 0.6
                r = int(60 + (255 - 60) * t)
                g = int(190 + (215 - 190) * t)
                b = int(90 - 70 * t)
            else:
                t = (norm - 0.6) / 0.4
                r = 255
                g = int(215 - 155 * t)
                b = int(20 * (1 - t))
            return f"#{r:02x}{g:02x}{b:02x}"

        def redraw(self, norm: float) -> None:
            self.delete("all")
            w = max(80, self.winfo_width())
            h = max(24, self.winfo_height())
            self.create_rectangle(0, 0, w, h, fill="#111111", outline="")
            self.create_text(8, h / 2, anchor="w", text=self.title, fill="#d0d0d0", font=("TkDefaultFont", 8, "bold"))
            bar_x0 = 58
            bar_x1 = w - 8
            bar_y0 = 6
            bar_y1 = h - 6
            segments = 28
            gap = 2
            seg_w = max(2, (bar_x1 - bar_x0 - gap * (segments - 1)) / segments)
            norm = max(0.0, min(1.0, norm))
            active = int(round(norm * segments))
            for i in range(segments):
                x0 = bar_x0 + i * (seg_w + gap)
                x1 = x0 + seg_w
                fill = self._color_for_norm((i + 1) / segments) if i < active else "#262626"
                outline = "#303030" if i >= active else fill
                self.create_rectangle(x0, bar_y0, x1, bar_y1, fill=fill, outline=outline)

    class AbletonSlider(tk.Canvas):
        def __init__(self, master, label: str, var, vmin: float, vmax: float, fmt="{:.2f}",
                     command=None, width=40, height=150, accent="#ff9f1c"):
            super().__init__(master, width=width, height=height + 26, bg="#1c1c1c", highlightthickness=0)
            self.label = label
            self.var = var
            self.vmin = vmin
            self.vmax = vmax
            self.fmt = fmt
            self.command = command
            self.w = width
            self.h = height
            self.track_top = 10
            self.track_bot = height - 10
            self.accent = accent
            self.bind("<Button-1>", self._click)
            self.bind("<B1-Motion>", self._click)
            self.bind("<MouseWheel>", self._wheel)
            self.bind("<Button-4>", lambda e: self._step(1))
            self.bind("<Button-5>", lambda e: self._step(-1))
            try:
                self.var.trace_add("write", lambda *_: self.redraw())
            except Exception:
                pass
            self.redraw()

        def _get(self) -> float:
            try:
                return float(self.var.get())
            except Exception:
                return self.vmin

        def _set(self, val: float) -> None:
            val = max(self.vmin, min(self.vmax, val))
            if isinstance(self.var, tk.IntVar):
                self.var.set(int(round(val)))
            else:
                self.var.set(f"{val:.3f}".rstrip("0").rstrip("."))
            if self.command:
                self.command()
            self.redraw()

        def _step(self, direction: int) -> None:
            self._set(self._get() + direction * ((self.vmax - self.vmin) / 120.0))

        def _wheel(self, event):
            self._step(1 if event.delta > 0 else -1)

        def _click(self, event):
            y = min(max(event.y, self.track_top), self.track_bot)
            norm = 1.0 - ((y - self.track_top) / max(1, (self.track_bot - self.track_top)))
            self._set(self.vmin + norm * (self.vmax - self.vmin))

        def redraw(self):
            self.delete("all")
            val = self._get()
            norm = 0.0 if self.vmax == self.vmin else (val - self.vmin) / (self.vmax - self.vmin)
            norm = max(0.0, min(1.0, norm))
            x = self.w / 2
            self.create_rectangle(x - 7, self.track_top, x + 7, self.track_bot, fill="#0f0f0f", outline="#303030")
            y = self.track_top + (1.0 - norm) * (self.track_bot - self.track_top)
            self.create_rectangle(x - 7, y, x + 7, self.track_bot, fill=self.accent, outline="")
            self.create_rectangle(x - 14, y - 5, x + 14, y + 5, fill="#d6d6d6", outline="#777777")
            self.create_text(x, self.h + 4, text=self.label, fill="#d0d0d0", font=("TkDefaultFont", 7, "bold"))
            self.create_text(x, self.h + 16, text=self.fmt.format(val), fill="#9aa0a6", font=("TkDefaultFont", 7))

    root = tk.Tk()
    root.title("Voice Bridge")
    root.geometry("360x360")
    root.minsize(340, 330)
    root.configure(bg="#181818")

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass
    style.configure(".", background="#181818", foreground="#d0d0d0", fieldbackground="#232323")
    style.configure("TFrame", background="#181818")
    style.configure("TLabel", background="#181818", foreground="#d0d0d0")
    style.configure("TLabelframe", background="#181818", foreground="#f5f5f5")
    style.configure("TLabelframe.Label", background="#181818", foreground="#f5f5f5", font=("TkDefaultFont", 9, "bold"))
    style.configure("TButton", background="#2b2b2b", foreground="#f0f0f0")
    style.configure("TCheckbutton", background="#181818", foreground="#d0d0d0")
    style.configure("TCombobox", fieldbackground="#232323", foreground="#d0d0d0")
    style.configure("Mini.TButton", padding=(6, 2))

    bridge_ref = {"bridge": None}
    current_config_path = {"path": default_args.config or ""}
    advanced_open = {"win": None}

    vars_map = {
        "peer_host": tk.StringVar(value=default_args.peer_host),
        "peer_port": tk.StringVar(value=str(default_args.peer_port)),
        "listen_port": tk.StringVar(value=str(default_args.listen_port)),
        "input_device_name": tk.StringVar(value=default_args.input_device_name or ""),
        "output_device_name": tk.StringVar(value=default_args.output_device_name or ""),
        "input_device": tk.StringVar(value="" if default_args.input_device is None else str(default_args.input_device)),
        "output_device": tk.StringVar(value="" if default_args.output_device is None else str(default_args.output_device)),
        "aec_method": tk.StringVar(value=default_args.aec_method),

        "opus_bitrate": tk.StringVar(value=str(default_args.opus_bitrate)),
        "input_gain": tk.StringVar(value=str(default_args.input_gain)),
        "output_gain": tk.StringVar(value=str(default_args.output_gain)),
        "vad_threshold": tk.StringVar(value=str(default_args.vad_threshold)),
        "recv_queue_frames": tk.StringVar(value=str(default_args.recv_queue_frames)),

        "opus_frame_ms": tk.StringVar(value=str(default_args.opus_frame_ms)),
        "opus_complexity": tk.StringVar(value=str(default_args.opus_complexity)),
        "vad_hangover_frames": tk.StringVar(value=str(default_args.vad_hangover_frames)),
        "preemphasis": tk.StringVar(value=str(default_args.preemphasis)),
        "dsp_lowpass_alpha": tk.StringVar(value=str(default_args.dsp_lowpass_alpha)),
        "dsp_output_limit": tk.StringVar(value=str(default_args.dsp_output_limit)),

        "dsp_noise_gate_enable": tk.BooleanVar(value=not default_args.dsp_noise_gate_disable),
        "dsp_lowpass_enable": tk.BooleanVar(value=not default_args.dsp_lowpass_disable),
        "dsp_output_limiter_enable": tk.BooleanVar(value=not default_args.dsp_output_limiter_disable),
        "dsp_dc_block": tk.BooleanVar(value=not default_args.dsp_no_dc_block),
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
        inputs = [""]
        outputs = [""]
        for i, dev in enumerate(devs):
            label = f"[{i}] {dev['name']}"
            if int(dev.get("max_input_channels", 0)) > 0:
                inputs.append(label)
            if int(dev.get("max_output_channels", 0)) > 0:
                outputs.append(label)
        input_combo.configure(values=inputs)
        output_combo.configure(values=outputs)
        if advanced_open["win"] is not None and advanced_open["win"].winfo_exists():
            try:
                adv_input_combo.configure(values=inputs)
                adv_output_combo.configure(values=outputs)
            except Exception:
                pass
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
        return None if not s else int(s)

    def build_config() -> AudioConfig:
        return AudioConfig(
            peer_host=vars_map["peer_host"].get().strip(),
            peer_port=int(vars_map["peer_port"].get()),
            listen_port=int(vars_map["listen_port"].get()),
            input_device=int_or_none(vars_map["input_device"].get()),
            output_device=int_or_none(vars_map["output_device"].get()),
            input_device_name=vars_map["input_device_name"].get().strip() or None,
            output_device_name=vars_map["output_device_name"].get().strip() or None,
            aec_mode="complete",
            aec_method=vars_map["aec_method"].get(),
            opus_bitrate=int(float(vars_map["opus_bitrate"].get())),
            opus_frame_ms=int(float(vars_map["opus_frame_ms"].get())),
            opus_complexity=int(float(vars_map["opus_complexity"].get())),
            input_gain=float(vars_map["input_gain"].get()),
            output_gain=float(vars_map["output_gain"].get()),
            vad_threshold=int(float(vars_map["vad_threshold"].get())),
            recv_queue_frames=int(float(vars_map["recv_queue_frames"].get())),
            vad_hangover_frames=int(float(vars_map["vad_hangover_frames"].get())),
            preemphasis=float(vars_map["preemphasis"].get()),
            dsp_lowpass_alpha=float(vars_map["dsp_lowpass_alpha"].get()),
            dsp_output_limit=float(vars_map["dsp_output_limit"].get()),
            dsp_noise_gate_enable=bool(vars_map["dsp_noise_gate_enable"].get()),
            dsp_lowpass_enable=bool(vars_map["dsp_lowpass_enable"].get()),
            dsp_output_limiter_enable=bool(vars_map["dsp_output_limiter_enable"].get()),
            dsp_dc_block=bool(vars_map["dsp_dc_block"].get()),
            dsp_echo_reduce_enable=False,
        )

    def apply_live_settings() -> None:
        bridge = bridge_ref.get("bridge")
        if bridge is None:
            return
        try:
            bridge.apply_live_settings(build_config())
        except Exception as e:
            log(f"Live update failed: {e}")

    def save_yaml() -> None:
        try:
            cfg = build_config()
        except Exception as e:
            messagebox.showerror("Save failed", str(e))
            return
        path = filedialog.asksaveasfilename(title="Save YAML", defaultextension=".yaml",
                                            filetypes=[("YAML", "*.yaml *.yml"), ("All files", "*.*")])
        if not path:
            return
        save_config_yaml(path, asdict(cfg))
        current_config_path["path"] = path
        config_path_var.set(path)
        log(f"Config saved: {path}")

    def load_yaml() -> None:
        path = filedialog.askopenfilename(title="Load YAML", filetypes=[("YAML", "*.yaml *.yml"), ("All files", "*.*")])
        if not path:
            return
        try:
            data = load_config_yaml(path)
            for key, value in data.items():
                if key in vars_map:
                    var = vars_map[key]
                    if isinstance(var, tk.BooleanVar):
                        var.set(bool(value))
                    else:
                        var.set("" if value is None else str(value))
            current_config_path["path"] = path
            config_path_var.set(path)
            log(f"Config loaded: {path}")
            apply_live_settings()
        except Exception as e:
            messagebox.showerror("Load failed", str(e))

    def apply_preset(kind: str) -> None:
        presets = {
            "Low latency": {"opus_bitrate": "10000", "vad_threshold": "600", "recv_queue_frames": "3", "opus_frame_ms": "20", "opus_complexity": "3"},
            "Balanced": {"opus_bitrate": "12000", "vad_threshold": "500", "recv_queue_frames": "4", "opus_frame_ms": "20", "opus_complexity": "4"},
            "Stable": {"opus_bitrate": "14000", "vad_threshold": "450", "recv_queue_frames": "6", "opus_frame_ms": "40", "opus_complexity": "5"},
        }
        data = presets.get(kind)
        if not data:
            return
        for k, v in data.items():
            vars_map[k].set(v)
        apply_live_settings()
        log(f"Preset applied: {kind}")

    def start_bridge() -> None:
        if bridge_ref["bridge"] is not None:
            return
        try:
            bridge = AudioBridge(build_config(), logger=log)
            bridge.start()
            bridge_ref["bridge"] = bridge
            start_btn.configure(state="disabled")
            stop_btn.configure(state="normal")
            test_btn.configure(state="normal")
        except Exception as e:
            messagebox.showerror("Start failed", str(e))

    def stop_bridge() -> None:
        bridge = bridge_ref.get("bridge")
        if bridge is None:
            return
        try:
            bridge.stop()
        finally:
            bridge_ref["bridge"] = None
            start_btn.configure(state="normal")
            stop_btn.configure(state="disabled")
            test_btn.configure(state="disabled")

    def test_connection() -> None:
        bridge = bridge_ref.get("bridge")
        if bridge is not None:
            bridge.send_test()

    def refresh_status() -> None:
        bridge = bridge_ref.get("bridge")
        if bridge is None:
            status_var.set("stopped")
            mic_meter.redraw(0.0)
            remote_meter.redraw(0.0)
        else:
            status_var.set(bridge.status_text())
            vis = bridge.get_visual_state()
            mic_meter.redraw(min(1.0, vis["input_rms"] * 4.0))
            remote_meter.redraw(min(1.0, vis["rx_rms"] * 4.0))
        root.after(120, refresh_status)

    def open_advanced() -> None:
        if advanced_open["win"] is not None and advanced_open["win"].winfo_exists():
            advanced_open["win"].lift()
            return

        win = tk.Toplevel(root)
        advanced_open["win"] = win
        win.title("Advanced")
        win.geometry("560x500")
        win.configure(bg="#181818")

        frm = ttk.Frame(win, padding=8)
        frm.pack(fill="both", expand=True)
        frm.columnconfigure(0, weight=1)
        frm.columnconfigure(1, weight=1)

        topf = ttk.LabelFrame(frm, text="Routing")
        topf.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        topf.columnconfigure(1, weight=1)
        ttk.Label(topf, text="Input").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=4)
        global adv_input_combo
        adv_input_combo = ttk.Combobox(topf, textvariable=vars_map["input_device_name"], width=20)
        adv_input_combo.grid(row=0, column=1, sticky="ew", pady=4)
        adv_input_combo.bind("<<ComboboxSelected>>", lambda _e: sync_name_to_id("input"))
        ttk.Label(topf, text="Output").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=4)
        global adv_output_combo
        adv_output_combo = ttk.Combobox(topf, textvariable=vars_map["output_device_name"], width=20)
        adv_output_combo.grid(row=1, column=1, sticky="ew", pady=4)
        adv_output_combo.bind("<<ComboboxSelected>>", lambda _e: sync_name_to_id("output"))
        ttk.Label(topf, text="AEC backend").grid(row=2, column=0, sticky="w", padx=(0, 6), pady=4)
        ttk.Combobox(topf, textvariable=vars_map["aec_method"], values=["webrtc", "speex"], state="readonly", width=18).grid(row=2, column=1, sticky="ew", pady=4)

        slidef = ttk.LabelFrame(frm, text="Advanced sliders")
        slidef.grid(row=1, column=0, columnspan=2, sticky="nsew")
        slidef.columnconfigure(0, weight=1)
        sliders = ttk.Frame(slidef)
        sliders.pack(fill="both", expand=True, padx=6, pady=6)

        AbletonSlider(sliders, "BIT", vars_map["opus_bitrate"], 4000, 32000, fmt="{:.0f}", command=apply_live_settings, accent="#8e8eff").pack(side="left", padx=8)
        AbletonSlider(sliders, "FRAME", vars_map["opus_frame_ms"], 20, 60, fmt="{:.0f}", command=apply_live_settings, accent="#8e8eff").pack(side="left", padx=8)
        AbletonSlider(sliders, "COMP", vars_map["opus_complexity"], 0, 10, fmt="{:.0f}", command=apply_live_settings, accent="#8e8eff").pack(side="left", padx=8)
        AbletonSlider(sliders, "HANG", vars_map["vad_hangover_frames"], 0, 10, fmt="{:.0f}", command=apply_live_settings, accent="#8e8eff").pack(side="left", padx=8)
        AbletonSlider(sliders, "PRE", vars_map["preemphasis"], 0.0, 1.0, fmt="{:.2f}", command=apply_live_settings, accent="#8e8eff").pack(side="left", padx=8)
        AbletonSlider(sliders, "LPF", vars_map["dsp_lowpass_alpha"], 0.01, 1.0, fmt="{:.2f}", command=apply_live_settings, accent="#8e8eff").pack(side="left", padx=8)
        AbletonSlider(sliders, "LIMIT", vars_map["dsp_output_limit"], 0.10, 1.0, fmt="{:.2f}", command=apply_live_settings, accent="#8e8eff").pack(side="left", padx=8)
        AbletonSlider(sliders, "QUEUE", vars_map["recv_queue_frames"], 2, 12, fmt="{:.0f}", command=apply_live_settings, accent="#8e8eff").pack(side="left", padx=8)

        tog = ttk.LabelFrame(frm, text="DSP toggles")
        tog.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Checkbutton(tog, text="DC block", variable=vars_map["dsp_dc_block"], command=apply_live_settings).pack(anchor="w", padx=8, pady=2)
        ttk.Checkbutton(tog, text="Noise gate", variable=vars_map["dsp_noise_gate_enable"], command=apply_live_settings).pack(anchor="w", padx=8, pady=2)
        ttk.Checkbutton(tog, text="Low-pass", variable=vars_map["dsp_lowpass_enable"], command=apply_live_settings).pack(anchor="w", padx=8, pady=2)
        ttk.Checkbutton(tog, text="Output limiter", variable=vars_map["dsp_output_limiter_enable"], command=apply_live_settings).pack(anchor="w", padx=8, pady=2)

        note = ttk.Label(frm, text="Backend, device, frame, complexity, and queue changes are safest after reconnect.")
        note.grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))

        refresh_devices()

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    main = ttk.Frame(root, padding=8)
    main.grid(row=0, column=0, sticky="nsew")
    main.columnconfigure(0, weight=0)
    main.columnconfigure(1, weight=1)
    main.rowconfigure(1, weight=1)

    top = ttk.Frame(main)
    top.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
    for c in range(4):
        top.columnconfigure(c, weight=1)
    start_btn = ttk.Button(top, text="Connect", command=start_bridge, style="Mini.TButton")
    stop_btn = ttk.Button(top, text="Disconnect", command=stop_bridge, state="disabled", style="Mini.TButton")
    test_btn = ttk.Button(top, text="Ping", command=test_connection, state="disabled", style="Mini.TButton")
    adv_btn = ttk.Button(top, text="Advanced", command=open_advanced, style="Mini.TButton")
    preset_var = tk.StringVar(value="Balanced")
    preset_box = ttk.Combobox(top, textvariable=preset_var, values=["Low latency", "Balanced", "Stable"], state="readonly", width=10)
    preset_apply = ttk.Button(top, text="Apply", command=lambda: apply_preset(preset_var.get()), style="Mini.TButton")
    load_btn = ttk.Button(top, text="Load", command=load_yaml, style="Mini.TButton")
    save_btn = ttk.Button(top, text="Save", command=save_yaml, style="Mini.TButton")
    widgets = [start_btn, stop_btn, test_btn, adv_btn, preset_box, preset_apply, load_btn, save_btn]
    for i, w in enumerate(widgets):
        r, c = divmod(i, 4)
        w.grid(row=r, column=c, sticky="ew", padx=2, pady=2)

    body = ttk.Frame(main)
    body.grid(row=1, column=0, columnspan=2, sticky="nsew")
    body.columnconfigure(0, weight=1)
    body.columnconfigure(1, weight=1)
    body.rowconfigure(2, weight=1)

    session = ttk.LabelFrame(body, text="Session")
    session.grid(row=0, column=0, sticky="nsew", padx=(0, 4), pady=(0, 4))
    session.columnconfigure(1, weight=1)
    for r, (label, key) in enumerate([("IP", "peer_host"), ("Send", "peer_port"), ("Recv", "listen_port")]):
        ttk.Label(session, text=label).grid(row=r, column=0, sticky="w", padx=(0, 6), pady=1)
        ttk.Entry(session, textvariable=vars_map[key], width=11).grid(row=r, column=1, sticky="ew", pady=1)

    devices = ttk.LabelFrame(body, text="Devices")
    devices.grid(row=0, column=1, sticky="nsew", padx=(4, 0), pady=(0, 4))
    devices.columnconfigure(1, weight=1)
    ttk.Label(devices, text="Input").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=1)
    input_combo = ttk.Combobox(devices, textvariable=vars_map["input_device_name"], width=13)
    input_combo.grid(row=0, column=1, sticky="ew", pady=1)
    ttk.Label(devices, text="Output").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=1)
    output_combo = ttk.Combobox(devices, textvariable=vars_map["output_device_name"], width=13)
    output_combo.grid(row=1, column=1, sticky="ew", pady=1)
    input_combo.bind("<<ComboboxSelected>>", lambda _e: sync_name_to_id("input"))
    output_combo.bind("<<ComboboxSelected>>", lambda _e: sync_name_to_id("output"))

    status_var = tk.StringVar(value="stopped")

    ctrl = ttk.LabelFrame(body, text="Volume")
    ctrl.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 4))
    sliders = ttk.Frame(ctrl)
    sliders.pack(fill="x", padx=6, pady=4)
    AbletonSlider(sliders, "IN", vars_map["input_gain"], 0.2, 3.0, fmt="{:.2f}", command=apply_live_settings).pack(side="left", padx=8)
    AbletonSlider(sliders, "OUT", vars_map["output_gain"], 0.2, 3.0, fmt="{:.2f}", command=apply_live_settings).pack(side="left", padx=8)
    status_line = ttk.Frame(ctrl)
    status_line.pack(fill="x", padx=6, pady=(0, 4))
    ttk.Label(status_line, text="AEC: complete").pack(side="left")
    ttk.Label(status_line, textvariable=status_var).pack(side="left", padx=(10, 0))

    meters = ttk.LabelFrame(body, text="Levels")
    meters.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(0, 0))
    meters.columnconfigure(0, weight=1)
    mic_meter = MeterBar(meters, "Mic", height=30)
    remote_meter = MeterBar(meters, "Remote", height=30)
    mic_meter.grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 3))
    remote_meter.grid(row=1, column=0, sticky="ew", padx=8, pady=(3, 6))

    text = scrolledtext.ScrolledText(body, height=2, bg="#101010", fg="#d0d0d0", insertbackground="#ffffff")
    text.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(4, 0))
    text.configure(state="disabled")

    refresh_devices()
    refresh_status()
    root.protocol("WM_DELETE_WINDOW", lambda: (stop_bridge(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    args = parse_args()
    args = apply_config_to_args(args)
    if args.gui or len(__import__("sys").argv) == 1:
        run_gui(args)
    else:
        raise SystemExit(run_cli(args))
