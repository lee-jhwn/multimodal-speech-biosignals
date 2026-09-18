import argparse
import os
import re
import shutil
import subprocess
from typing import List, Tuple

import cv2
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import mne
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cut run video by end-aligned VMRK timing and place it next to trigger topomaps."
        )
    )
    parser.add_argument(
        "--vmrk-path",
        default="/Volumes/T9/lambda43_backup/jihwan/projects/2601_eegmri/data/recovered_triggers/251121_2_scanneron_phonated_1_fixfixfix_CCA_Cleaned.vmrk",
    )
    parser.add_argument(
        "--vhdr-path",
        default="/Volumes/T9/lambda43_backup/jihwan/projects/2601_eegmri/data/recovered_triggers/251121_2_scanneron_phonated_1_fixfixfix_CCA_Cleaned.vhdr",
    )
    parser.add_argument(
        "--video-path",
        default="/Volumes/T9/lambda43_backup/jihwan/projects/2601_eegmri/data/vol1395_20251121/video/stcr/video_with_audio_correct_names/avi_with_audio/251121_2_scanneron_phonated_1_fixfixfix.avi",
    )
    parser.add_argument(
        "--evoked-path",
        default="/Volumes/T9/lambda43_backup/jihwan/projects/2601_eegmri/data/for_icassp_talk/trigger57_grand_average-ave.fif",
    )
    parser.add_argument(
        "--output-dir",
        default="/Volumes/T9/lambda43_backup/jihwan/projects/2601_eegmri/data/for_icassp_talk",
    )
    parser.add_argument("--trigger", type=int, default=57)
    parser.add_argument("--tmin", type=float, default=-1.5)
    parser.add_argument("--tmax", type=float, default=2.5)
    parser.add_argument("--fps", type=float, default=9.0)
    parser.add_argument("--vmin-uv", type=float, default=-24.0)
    parser.add_argument("--vmax-uv", type=float, default=24.0)
    parser.add_argument(
        "--trigger-occurrence",
        choices=["first", "last"],
        default="last",
        help="Which occurrence of trigger to align to in the marker file.",
    )
    return parser.parse_args()


def parse_vmrk_positions(vmrk_path: str, trigger_code: int) -> Tuple[List[int], List[int]]:
    trigger_positions: List[int] = []
    volume_positions: List[int] = []

    mk_re = re.compile(r"^Mk\d+=(?P<type>[^,]*),(?P<desc>[^,]*),(?P<pos>\d+),")
    trigger_re = re.compile(rf"\bS\s*{trigger_code}\b")

    with open(vmrk_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            m = mk_re.match(line)
            if not m:
                continue
            desc = m.group("desc")
            pos = int(m.group("pos"))

            if "Volume/V" in desc:
                volume_positions.append(pos)
            if trigger_re.search(desc):
                trigger_positions.append(pos)

    return trigger_positions, volume_positions


def figure_to_rgb_array(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return rgba[:, :, :3].copy()


def make_topomap_frame(
    evoked: mne.Evoked,
    frame_idx: int,
    trigger_code: int,
    vlim_uv: Tuple[float, float],
) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=120)
    mne.viz.plot_topomap(
        evoked.data[:, frame_idx] * 1e6,
        evoked.info,
        axes=ax,
        show=False,
        cmap="RdBu_r",
        vlim=vlim_uv,
        contours=0,
        sensors=True,
    )

    t = evoked.times[frame_idx]
    ax.text(
        0.005,
        0.995,
        f"t={t:+.3f}s",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        color="black",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2.0),
    )
    frame = figure_to_rgb_array(fig)
    plt.close(fig)
    return frame


def get_video_info(video_path: str) -> Tuple[float, int, int, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if fps <= 0 or frame_count <= 0:
        raise RuntimeError("Video metadata invalid (fps or frame_count <= 0).")

    duration = frame_count / fps
    return fps, width, height, duration


def read_video_frame_at_time(cap: cv2.VideoCapture, t_sec: float, fps: float) -> np.ndarray:
    frame_idx = int(round(t_sec * fps))
    frame_idx = max(0, frame_idx)
    max_idx = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    frame_idx = min(frame_idx, max_idx)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame_bgr = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read video frame at index {frame_idx}")

    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == target_h:
        return img
    scale = target_h / float(h)
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)


def make_stimulus_panel(height: int, width: int, rel_t: float) -> np.ndarray:
    panel = np.zeros((height, width, 3), dtype=np.uint8)

    def draw_cross(img: np.ndarray) -> None:
        cy = img.shape[0] // 2
        cx = img.shape[1] // 2
        size = max(10, min(img.shape[0], img.shape[1]) // 8)
        thickness = max(2, min(img.shape[0], img.shape[1]) // 80)
        cv2.line(img, (cx - size, cy), (cx + size, cy), (255, 255, 255), thickness)
        cv2.line(img, (cx, cy - size), (cx, cy + size), (255, 255, 255), thickness)

    if -1.5 <= rel_t < -1.0:
        draw_cross(panel)
    elif -1.0 <= rel_t < 0.0:
        text = "ana"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.8, height / 420.0)
        thickness = max(2, int(round(font_scale * 2)))
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x = (width - tw) // 2
        y = (height + th) // 2
        cv2.putText(panel, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    elif 0.0 <= rel_t < 2.0:
        draw_cross(panel)
    # After 2.0 sec, panel remains black.

    return panel


def overlay_stimulus_panel(
    combined: np.ndarray,
    topo_width: int,
    panel_width: int,
    panel_height: int,
    rel_t: float,
) -> np.ndarray:
    h, w = combined.shape[:2]
    panel = make_stimulus_panel(panel_height, panel_width, rel_t)

    # Center panel on topo-video boundary and anchor it to the bottom.
    boundary_x = topo_width
    x0 = boundary_x - panel_width // 2
    y0 = h - panel_height

    x0 = max(0, min(x0, w - panel_width))
    y0 = max(0, min(y0, h - panel_height))

    out = combined.copy()
    out[y0 : y0 + panel_height, x0 : x0 + panel_width] = panel
    return out


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    raw = mne.io.read_raw_brainvision(args.vhdr_path, preload=False, verbose=False)
    sfreq = float(raw.info["sfreq"])

    trigger_positions, volume_positions = parse_vmrk_positions(args.vmrk_path, args.trigger)
    if not trigger_positions:
        raise RuntimeError(f"No S {args.trigger} markers found in {args.vmrk_path}")
    if not volume_positions:
        raise RuntimeError(f"No Volume/V markers found in {args.vmrk_path}")

    trigger_pos = trigger_positions[0] if args.trigger_occurrence == "first" else trigger_positions[-1]
    last_volume_pos = volume_positions[-1]

    # Convert VMRK point positions to seconds (1-based points in BrainVision markers).
    trigger_sec = (trigger_pos - 1) / sfreq
    last_volume_sec = (last_volume_pos - 1) / sfreq
    delta_to_end_sec = last_volume_sec - trigger_sec

    video_fps, _, _, video_duration = get_video_info(args.video_path)
    trigger_video_sec = video_duration - delta_to_end_sec

    cut_start = trigger_video_sec + args.tmin
    cut_end = trigger_video_sec + args.tmax
    if cut_start < 0 or cut_end > video_duration:
        raise RuntimeError(
            f"Requested cut [{cut_start:.3f}, {cut_end:.3f}] is outside video duration [0, {video_duration:.3f}]"
        )

    evoked = mne.read_evokeds(args.evoked_path, condition=0, verbose=False)
    ev_anim = evoked.copy().resample(args.fps, npad="auto")

    n_frames = len(ev_anim.times)
    expected_frames = int(round((args.tmax - args.tmin) * args.fps))
    if n_frames != expected_frames:
        # Keep the actual evoked frame count; this can differ by one due to sample boundaries.
        print(f"Frame count note: expected {expected_frames}, using {n_frames}")

    vlim_uv = (args.vmin_uv, args.vmax_uv)
    fps_label = f"{int(args.fps)}fps" if float(args.fps).is_integer() else f"{args.fps}fps"
    base = f"trigger{args.trigger}_topomap_plus_video_{fps_label}"
    out_mp4_noaudio = os.path.join(args.output_dir, f"{base}_noaudio.mp4")
    out_mp4 = os.path.join(args.output_dir, f"{base}.mp4")
    out_gif = os.path.join(args.output_dir, f"{base}.gif")
    out_meta = os.path.join(args.output_dir, f"{base}_metadata.txt")

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video_path}")

    mp4_writer = imageio.get_writer(
        out_mp4_noaudio,
        fps=args.fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    gif_writer = imageio.get_writer(out_gif, mode="I", fps=args.fps, loop=0)

    try:
        for i in range(n_frames):
            if i % 10 == 0 or i == n_frames - 1:
                print(f"Composing frame {i + 1}/{n_frames}")

            rel_t = args.tmin + (i / args.fps)
            video_t = trigger_video_sec + rel_t

            topo = make_topomap_frame(ev_anim, i, args.trigger, vlim_uv)
            video = read_video_frame_at_time(cap, video_t, video_fps)
            video = resize_to_height(video, topo.shape[0])

            combined = np.concatenate([topo, video], axis=1)
            stim_width = max(120, topo.shape[1] // 5)
            stim_height = max(90, topo.shape[0] // 4)
            combined = overlay_stimulus_panel(
                combined,
                topo_width=topo.shape[1],
                panel_width=stim_width,
                panel_height=stim_height,
                rel_t=rel_t,
            )
            gif_writer.append_data(combined)
            mp4_writer.append_data(combined)
    finally:
        cap.release()
        gif_writer.close()
        mp4_writer.close()

    # Add aligned audio from the source video to the composed MP4.
    duration_sec = args.tmax - args.tmin
    ffmpeg = shutil.which("ffmpeg")
    audio_muxed = False
    ffmpeg_cmd = ""
    if ffmpeg:
        ffmpeg_cmd_list = [
            ffmpeg,
            "-y",
            "-ss",
            f"{cut_start:.6f}",
            "-t",
            f"{duration_sec:.6f}",
            "-i",
            args.video_path,
            "-i",
            out_mp4_noaudio,
            "-map",
            "1:v:0",
            "-map",
            "0:a:0?",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            out_mp4,
        ]
        ffmpeg_cmd = " ".join(ffmpeg_cmd_list)
        try:
            subprocess.run(ffmpeg_cmd_list, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            audio_muxed = True
        except subprocess.CalledProcessError:
            audio_muxed = False

    if not audio_muxed:
        # Fallback: keep the no-audio MP4 as the primary output.
        if os.path.exists(out_mp4):
            os.remove(out_mp4)
        os.replace(out_mp4_noaudio, out_mp4)
    else:
        if os.path.exists(out_mp4_noaudio):
            os.remove(out_mp4_noaudio)

    with open(out_meta, "w", encoding="utf-8") as f:
        f.write(f"vmrk_path={args.vmrk_path}\n")
        f.write(f"video_path={args.video_path}\n")
        f.write(f"evoked_path={args.evoked_path}\n")
        f.write(f"trigger={args.trigger}\n")
        f.write(f"trigger_occurrence={args.trigger_occurrence}\n")
        f.write(f"trigger_positions_count={len(trigger_positions)}\n")
        f.write(f"chosen_trigger_position={trigger_pos}\n")
        f.write(f"last_volume_position={last_volume_pos}\n")
        f.write(f"sfreq={sfreq}\n")
        f.write(f"trigger_sec_in_recording={trigger_sec:.6f}\n")
        f.write(f"last_volume_sec_in_recording={last_volume_sec:.6f}\n")
        f.write(f"delta_to_end_sec={delta_to_end_sec:.6f}\n")
        f.write(f"video_duration_sec={video_duration:.6f}\n")
        f.write(f"trigger_sec_in_video={trigger_video_sec:.6f}\n")
        f.write(f"cut_start_sec={cut_start:.6f}\n")
        f.write(f"cut_end_sec={cut_end:.6f}\n")
        f.write(f"tmin={args.tmin}\n")
        f.write(f"tmax={args.tmax}\n")
        f.write(f"fps={args.fps}\n")
        f.write(f"frame_count={n_frames}\n")
        f.write(f"vlim_uv={vlim_uv}\n")
        f.write(f"audio_muxed={audio_muxed}\n")
        f.write(f"ffmpeg_found={bool(ffmpeg)}\n")
        if ffmpeg_cmd:
            f.write(f"ffmpeg_cmd={ffmpeg_cmd}\n")

    print(f"Saved MP4: {out_mp4}")
    print(f"Saved GIF: {out_gif}")
    print(f"Saved metadata: {out_meta}")


if __name__ == "__main__":
    main()
