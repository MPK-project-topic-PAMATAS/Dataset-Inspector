"""Read-only parser for the TI AWR1843BOOST radar configuration used by the
Data-Collection-GCS project.

This module ONLY reads text files under Data-Collection-GCS -- it never
imports Data-Collection-GCS's own Python packages (which would pull in
hardware dependencies like pyserial and could have import-time side
effects), never executes any of its code, and never writes anything
anywhere. Every value it reports is either:

  * a literal taken verbatim from a specific file (a "configured" value),
  * a value computed from those literals via a documented formula
    (a "derived" value, always tagged as such), or
  * a static, human-authored note citing the exact file it came from,
    used only for known documentation caveats/conflicts discovered while
    inspecting the project (see KNOWN_CONFIGURATION_NOTES below) --
    never guessed.

If a file is missing or a value can't be parsed, the corresponding field
is left unavailable (None) rather than invented.
"""
from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# Sibling checkout next to Dataset-Inspector (override with DATA_COLLECTION_ROOT env var).
DATA_COLLECTION_ROOT = Path(
    os.environ.get("DATA_COLLECTION_ROOT")
    or Path(__file__).resolve().parents[2] / "Data-Collection-GCS"
)
RECEIVER_PY = DATA_COLLECTION_ROOT / "radar" / "receiver.py"
SETTINGS_PY = DATA_COLLECTION_ROOT / "config" / "settings.py"
POINT_CLOUD_DOC = DATA_COLLECTION_ROOT / "docs" / "RADAR_POINT_CLOUD.md"

_SPEED_OF_LIGHT_M_S = 299_792_458.0

# Known, directly-observed documentation/configuration conflicts found while
# inspecting Data-Collection-GCS for this dashboard. These are static notes
# citing exact files/lines read during that inspection -- not guesses, and
# not re-derived automatically (the underlying prose isn't structured data).
# See the session's implementation report for the full reasoning.
KNOWN_CONFIGURATION_NOTES = [
    (
        "radar/receiver.py (comment above _SENSOR_CONFIG_CMDS) states this "
        "firmware build cannot combine the 3TX chirp table with the "
        "range-Doppler heatmap output: 'This firmware build cannot combine "
        "the 3TX chirp table with the range-Doppler heatmap output "
        "(guiMonitor's heatmap flag) -- enabling both together makes "
        "sensorStart throw a firmware exception'. Consistent with this, the "
        "current guiMonitor line has its rangeDopplerHeatMap flag set to 0 "
        "(OFF)."
    ),
    (
        "docs/RADAR_POINT_CLOUD.md describes an experiment where the radar "
        "was 'briefly switched to a 3TX config to get real z values ... "
        "then reverted' back toward a 2TX+heatmap configuration, partly "
        "because of the exact 3TX/heatmap conflict above."
    ),
    (
        "CONFLICT: a recorded session under output_multi-record/ may contain "
        "BOTH a populated range-Doppler heatmap AND elevation (z) values that "
        "are all exactly 0.0 -- which is the opposite of what the *current* "
        "radar/receiver.py configuration would produce (heatmap OFF, "
        "elevation ON via 3TX). This suggests the configuration actually "
        "active when that recording was made does not match the "
        "configuration currently checked into radar/receiver.py. This "
        "dashboard reports the configuration as currently written in that "
        "file, alongside the actually-recorded data, without assuming "
        "either one is 'correct' for a given session -- compare them "
        "yourself in the CONFIG VS ACTUAL panel."
    ),
]


@dataclass
class ConfigValue:
    name: str
    value: Any
    unit: str = ""
    source: str = ""
    derived: bool = False
    formula: str = ""

    def display(self) -> str:
        if self.value is None:
            return "N/A / Not available"
        text = f"{self.value}{(' ' + self.unit) if self.unit else ''}"
        return text


@dataclass
class RadarConfigReport:
    available: bool = False
    errors: list = field(default_factory=list)
    sources_used: list = field(default_factory=list)
    raw_commands: list = field(default_factory=list)
    values: dict = field(default_factory=dict)  # name -> ConfigValue
    notes: list = field(default_factory=list)

    def get(self, name: str) -> Optional[ConfigValue]:
        return self.values.get(name)


def _extract_config_cmds(receiver_source: str) -> Optional[list]:
    match = re.search(r"_SENSOR_CONFIG_CMDS\s*=\s*(\[.*?\])", receiver_source, re.DOTALL)
    if not match:
        return None
    try:
        return ast.literal_eval(match.group(1))
    except (ValueError, SyntaxError):
        return None


def _find_cmd_tokens(cmds: list, keyword: str) -> Optional[list]:
    for cmd in cmds:
        parts = cmd.split()
        if parts and parts[0] == keyword:
            return parts[1:]
    return None


def _num(token: str):
    try:
        f = float(token)
        return int(f) if f.is_integer() else f
    except (TypeError, ValueError):
        return None


def _extract_settings_constants(settings_source: str) -> dict:
    values = {}
    for line in settings_source.splitlines():
        m = re.match(r"^\s*([A-Z][A-Z0-9_]*)\s*=\s*(.+?)\s*(#.*)?$", line)
        if not m:
            continue
        name, raw_value = m.group(1), m.group(2)
        try:
            values[name] = ast.literal_eval(raw_value)
        except (ValueError, SyntaxError):
            values[name] = raw_value.strip()
    return values


def load_radar_config(data_collection_root: Path = DATA_COLLECTION_ROOT) -> RadarConfigReport:
    """Reads (never executes/writes) the radar configuration from the
    Data-Collection-GCS project. Safe to call even if that project isn't
    present -- returns available=False with a clear error in that case.
    """
    report = RadarConfigReport()
    receiver_py = data_collection_root / "radar" / "receiver.py"
    settings_py = data_collection_root / "config" / "settings.py"

    if not receiver_py.exists():
        report.errors.append(f"radar configuration source not found: {receiver_py}")
        return report

    try:
        receiver_source = receiver_py.read_text()
    except OSError as e:
        report.errors.append(f"could not read {receiver_py}: {e}")
        return report

    report.sources_used.append(str(receiver_py))
    cmds = _extract_config_cmds(receiver_source)
    if cmds is None:
        report.errors.append(f"could not parse _SENSOR_CONFIG_CMDS in {receiver_py}")
        return report

    report.raw_commands = cmds
    report.available = True

    settings_values = {}
    if settings_py.exists():
        try:
            settings_values = _extract_settings_constants(settings_py.read_text())
            report.sources_used.append(str(settings_py))
        except OSError as e:
            report.errors.append(f"could not read {settings_py}: {e}")
    else:
        report.errors.append(f"{settings_py} not found (settings.py constants unavailable)")

    channel = _find_cmd_tokens(cmds, "channelCfg")
    adc = _find_cmd_tokens(cmds, "adcCfg")
    profile = _find_cmd_tokens(cmds, "profileCfg")
    frame = _find_cmd_tokens(cmds, "frameCfg")
    gui_monitor = _find_cmd_tokens(cmds, "guiMonitor")
    cfar = [c for c in cmds if c.startswith("cfarCfg")]

    def add(name, value, unit="", source="", derived=False, formula=""):
        report.values[name] = ConfigValue(name, value, unit, source, derived, formula)

    src_channel = f"{receiver_py.name}: channelCfg"
    src_adc = f"{receiver_py.name}: adcCfg"
    src_profile = f"{receiver_py.name}: profileCfg"
    src_frame = f"{receiver_py.name}: frameCfg"
    src_gui = f"{receiver_py.name}: guiMonitor"

    rx_mask = tx_mask = None
    if channel and len(channel) >= 2:
        rx_mask, tx_mask = _num(channel[0]), _num(channel[1])
        num_rx = bin(int(rx_mask)).count("1") if rx_mask is not None else None
        num_tx = bin(int(tx_mask)).count("1") if tx_mask is not None else None
        add("num_rx_antennas", num_rx, source=src_channel, derived=True, formula="popcount(rxMask)")
        add("num_tx_antennas", num_tx, source=src_channel, derived=True, formula="popcount(txMask)")
        if num_rx is not None and num_tx is not None:
            add("num_virtual_antennas", num_tx * num_rx, source=src_channel, derived=True, formula="numTx * numRx")

    if adc and len(adc) >= 2:
        adc_bits_code, adc_fmt_code = _num(adc[0]), _num(adc[1])
        bits_map = {0: 12, 1: 14, 2: 16}
        fmt_map = {0: "real", 1: "complex 1x (I/Q, filtered)", 2: "complex 2x (I/Q, unfiltered)"}
        add("adc_bits", bits_map.get(adc_bits_code), unit="bits", source=src_adc)
        add("adc_output_format", fmt_map.get(adc_fmt_code, str(adc_fmt_code)), source=src_adc)

    start_freq_ghz = idle_time_us = adc_start_us = ramp_end_us = None
    freq_slope = num_adc_samples = dig_out_rate_ksps = rx_gain_db = None
    if profile and len(profile) >= 14:
        (_profile_id, start_freq_ghz, idle_time_us, adc_start_us, ramp_end_us,
         _tx_power, _tx_phase, freq_slope, _tx_start, num_adc_samples,
         dig_out_rate_ksps, _hpf1, _hpf2, rx_gain_db) = (_num(t) for t in profile[:14])
        add("start_frequency", start_freq_ghz, unit="GHz", source=src_profile)
        add("idle_time", idle_time_us, unit="us", source=src_profile)
        add("adc_start_time", adc_start_us, unit="us", source=src_profile)
        add("ramp_end_time", ramp_end_us, unit="us", source=src_profile)
        add("freq_slope", freq_slope, unit="MHz/us", source=src_profile)
        add("num_adc_samples", num_adc_samples, source=src_profile)
        add("adc_sample_rate", dig_out_rate_ksps, unit="ksps", source=src_profile)
        add("rx_gain", rx_gain_db, unit="dB", source=src_profile)

    num_loops = frame_periodicity_ms = chirp_start_idx = chirp_end_idx = None
    if frame and len(frame) >= 7:
        chirp_start_idx, chirp_end_idx, num_loops, _num_frames, frame_periodicity_ms, _trig, _delay = (
            _num(t) for t in frame[:7]
        )
        add("num_loops", num_loops, source=src_frame)
        add("frame_periodicity", frame_periodicity_ms, unit="ms", source=src_frame)
        if frame_periodicity_ms:
            add("configured_frame_rate", 1000.0 / frame_periodicity_ms, unit="Hz",
                source=src_frame, derived=True, formula="1000 / framePeriodicity_ms")

    if gui_monitor and len(gui_monitor) >= 7:
        flags = ["detectedObjects", "logMagRange", "noiseProfile", "rangeAzimuthHeatmap",
                 "rangeDopplerHeatmap", "statsInfo"]
        gm_values = {name: _num(tok) for name, tok in zip(flags, gui_monitor[1:])}
        add("heatmap_output_enabled", bool(gm_values.get("rangeDopplerHeatmap")), source=src_gui)
        add("detected_objects_output_enabled", bool(gm_values.get("detectedObjects")), source=src_gui)

    if cfar:
        add("cfar_config_raw", cfar, source="receiver.py: cfarCfg (raw)")

    # --- derived RF/timing quantities -------------------------------- #
    if all(v is not None for v in (freq_slope, num_adc_samples, dig_out_rate_ksps)):
        sampling_time_s = num_adc_samples / (dig_out_rate_ksps * 1e3)
        bandwidth_hz = (freq_slope * 1e6 / 1e-6) * sampling_time_s
        range_res_m = _SPEED_OF_LIGHT_M_S / (2 * bandwidth_hz)
        add("bandwidth", bandwidth_hz / 1e9, unit="GHz", source=src_profile,
            derived=True, formula="freqSlope * (numAdcSamples / adcSampleRate)")
        add("range_resolution", range_res_m, unit="m", source=src_profile,
            derived=True, formula="c / (2 * bandwidth)")
        add("max_range", range_res_m * num_adc_samples, unit="m", source=src_profile,
            derived=True, formula="range_resolution * numAdcSamples")

    if all(v is not None for v in (idle_time_us, ramp_end_us, chirp_start_idx, chirp_end_idx, num_loops, start_freq_ghz)):
        chirps_per_burst = chirp_end_idx - chirp_start_idx + 1
        chirp_period_us = idle_time_us + ramp_end_us
        burst_period_us = chirp_period_us * chirps_per_burst
        wavelength_m = _SPEED_OF_LIGHT_M_S / (start_freq_ghz * 1e9)
        vel_res = wavelength_m / (2 * num_loops * burst_period_us * 1e-6)
        max_vel = wavelength_m / (4 * burst_period_us * 1e-6)
        add("chirps_per_burst", chirps_per_burst, source=src_frame, derived=True,
            formula="chirpEndIdx - chirpStartIdx + 1")
        add("velocity_resolution", vel_res, unit="m/s", source=src_profile,
            derived=True, formula="wavelength / (2 * numLoops * burstPeriod)")
        add("max_velocity", max_vel, unit="m/s", source=src_profile,
            derived=True, formula="wavelength / (4 * burstPeriod)")

    for settings_name in ("RADAR_NUM_RANGE_BINS", "RADAR_NUM_DOPPLER_BINS", "RADAR_MAX_DELTA_NS"):
        if settings_name in settings_values:
            add(settings_name.lower(), settings_values[settings_name], source=str(settings_py.name))

    report.notes = list(KNOWN_CONFIGURATION_NOTES)
    return report
