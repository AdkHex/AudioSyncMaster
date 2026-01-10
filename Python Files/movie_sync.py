import warnings

# Suppress specific warnings from libraries
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import soundfile as sf
import librosa
from scipy.signal import fftconvolve
import os
import subprocess
import argparse
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, Optional, List, Callable
try:
    from pymediainfo import MediaInfo
except ImportError:
    MediaInfo = None # type: ignore
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, SpinnerColumn

# Initialize Rich Console
console = Console()

def load_audio(path: str, sr: int, duration: Optional[float] = None, offset: float = 0, verbose: bool = False) -> Optional[np.ndarray]:
    """Loads audio from a file, handling video extraction via in-memory pipe."""
    video_exts = ['.mp4', '.mkv', '.webm', '.avi', '.mov', '.eac3', '.ac3']
    _, ext = os.path.splitext(path)
    is_video = ext.lower() in video_exts

    if not is_video:
        try:
            with sf.SoundFile(path, 'r') as f:
                seek_frame = int(offset * f.samplerate)
                f.seek(seek_frame)
                read_frames = -1 if duration is None else int(duration * f.samplerate)
                y = f.read(frames=read_frames, dtype='float32', always_2d=True)
                y = librosa.to_mono(y.T)
                if f.samplerate != sr:
                    y = librosa.resample(y, orig_sr=f.samplerate, target_sr=sr)
                return y
        except Exception:
            try:
                y, native_sr = librosa.load(path, sr=None, mono=True, duration=duration, offset=offset)
                if native_sr != sr:
                    y = librosa.resample(y, orig_sr=native_sr, target_sr=sr)
                return y
            except Exception as e_librosa:
                console.print(f"[yellow]Warning: Direct load failed for {os.path.basename(path)}: {e_librosa}. Trying FFmpeg.[/yellow]")

    try:
        cmd = ['ffmpeg']
        if offset > 0:
            cmd.extend(['-ss', str(offset)])
        cmd.extend(['-i', path])
        if duration is not None:
            cmd.extend(['-t', str(duration)])
        cmd.extend([
            '-vn', '-f', 's16le', '-acodec', 'pcm_s16le',
            '-ar', str(sr), '-ac', '1', '-'
        ])

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            if b"video stream" in stderr.lower() or b"doesn't contain any video stream" in stderr.lower():
                 y, _ = librosa.load(path, sr=sr, mono=True, duration=duration, offset=offset)
                 return y
            console.print(f"[red]Error: FFmpeg failed for {os.path.basename(path)}.[/red]")
            if verbose:
                console.print(f"[dim]{stderr.decode('utf-8', errors='ignore')}[/dim]")
            return None

        return np.frombuffer(stdout, dtype=np.int16).astype(np.float32) / 32768.0

    except FileNotFoundError:
        console.print("[red]Error: FFmpeg not found. Please ensure it's in your system's PATH.[/red]")
        return None
    except Exception as e:
        console.print(f"[red]An unexpected error occurred in load_audio for {os.path.basename(path)}: {e}[/red]")
        return None

def get_audio_duration(path: str, verbose: bool = False) -> Optional[float]:
    """Gets the duration of an audio or video file in seconds."""
    if MediaInfo:
        try:
            media_info = MediaInfo.parse(path)
            for track in media_info.tracks:
                if track.duration:
                    duration_s = float(track.duration) / 1000.0
                    if verbose: console.print(f"[dim]get_duration (mediainfo): {duration_s}s for {os.path.basename(path)}[/dim]")
                    return duration_s
        except Exception as e:
            if verbose: console.print(f"[dim]pymediainfo failed for {os.path.basename(path)}: {e}[/dim]")
    else:
        if verbose: console.print("[dim]pymediainfo library not installed, skipping.[/dim]")

    try:
        cmd = [
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', path
        ]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode == 0:
            duration_str = stdout.decode().strip()
            if duration_str and duration_str != "N/A":
                if verbose: console.print(f"[dim]get_duration (ffprobe): {duration_str}s for {os.path.basename(path)}[/dim]")
                return float(duration_str)
        if verbose:
            console.print(f"[dim]ffprobe failed for {os.path.basename(path)}. Stderr: {stderr.decode()}[/dim]")
    except FileNotFoundError:
        if verbose: console.print("[dim]ffprobe not found, skipping.[/dim]")
    except Exception as e:
        if verbose: console.print(f"[dim]ffprobe error for {os.path.basename(path)}: {e}[/dim]")

    try:
        duration = librosa.get_duration(path=path)
        if verbose: console.print(f"[dim]get_duration (librosa): {duration}s for {os.path.basename(path)}[/dim]")
        return duration
    except Exception as e:
        console.print(f"[red]Could not get duration for {os.path.basename(path)} with any method.[/red]")
        if verbose: console.print(f"[dim]Librosa error: {e}[/dim]")
        return None

def estimate_sync_offset_crosscorr(
    primary_audio: np.ndarray, secondary_audio: np.ndarray, sr: int
) -> float:
    """Estimate delay (in ms) using fast FFT cross-correlation."""
    def _normalize(y: np.ndarray) -> np.ndarray:
        y -= np.mean(y)
        std = np.std(y)
        if std > 1e-8:
            y /= std
        return y

    y_p = _normalize(primary_audio)
    y_s = _normalize(secondary_audio)

    corr = fftconvolve(y_p, y_s[::-1], mode='full')
    lag = np.argmax(corr) - (len(y_s) - 1)
    delay_sec = lag / sr
    return delay_sec * 1000

def process_pair(
    video_path: str, audio_path: str, segment_sec: float, verbose: bool = False,
    progress_callback: Optional[Callable[[int], None]] = None
) -> Tuple[str, str, Optional[float], Optional[float], Optional[str]]:
    """Processes a single video file against the audio file."""
    fast_sr = 8000
    start_delay: Optional[float] = None
    end_delay: Optional[float] = None

    try:
        # Start analysis
        video_audio_start = load_audio(video_path, sr=fast_sr, duration=segment_sec, verbose=verbose)
        if video_audio_start is None:
            return video_path, audio_path, None, None, f"Failed to load start of video: {os.path.basename(video_path)}"

        secondary_audio_start = load_audio(audio_path, sr=fast_sr, duration=segment_sec, verbose=verbose)
        if secondary_audio_start is None:
            return video_path, audio_path, None, None, f"Failed to load start of audio: {os.path.basename(audio_path)}"

        min_len_start = min(len(video_audio_start), len(secondary_audio_start))
        if min_len_start > fast_sr:
            start_delay = estimate_sync_offset_crosscorr(
                video_audio_start[:min_len_start], secondary_audio_start[:min_len_start], sr=fast_sr
            )
            if progress_callback:
                progress_callback(50)
        else:
            return video_path, audio_path, None, None, "Insufficient audio at start for analysis."

        # End analysis
        video_duration = get_audio_duration(video_path, verbose=verbose)
        audio_duration = get_audio_duration(audio_path, verbose=verbose)

        if video_duration is None or audio_duration is None:
            return video_path, audio_path, start_delay, None, "Could not get duration for end analysis."

        video_offset = max(0, video_duration - segment_sec)
        audio_offset = max(0, audio_duration - segment_sec)

        video_audio_end = load_audio(video_path, sr=fast_sr, duration=segment_sec, offset=video_offset, verbose=verbose)
        if video_audio_end is None:
            return video_path, audio_path, start_delay, None, f"Failed to load end of video: {os.path.basename(video_path)}"

        secondary_audio_end = load_audio(audio_path, sr=fast_sr, duration=segment_sec, offset=audio_offset, verbose=verbose)
        if secondary_audio_end is None:
            return video_path, audio_path, start_delay, None, f"Failed to load end of audio: {os.path.basename(audio_path)}"

        min_len_end = min(len(video_audio_end), len(secondary_audio_end))

        if min_len_end > fast_sr:
            end_delay_raw = estimate_sync_offset_crosscorr(
                video_audio_end[:min_len_end], secondary_audio_end[:min_len_end], sr=fast_sr
            )
            duration_diff_ms = (video_duration - audio_duration) * 1000
            end_delay = end_delay_raw + duration_diff_ms
        else:
            return video_path, audio_path, start_delay, None, "Insufficient audio at end for analysis."

        return (video_path, audio_path, start_delay, end_delay, None)

    except Exception as e:
        return (video_path, audio_path, start_delay, end_delay, str(e))

def main():
    """Main function to run the movie audio sync script."""
    try:
        console.print("[bold green]Movie Audio Sync Script started.[/bold green]")
        parser = argparse.ArgumentParser(
            description="Movie audio synchronization script - matches one audio file against multiple video files.",
            formatter_class=argparse.RawTextHelpFormatter
        )

        parser.add_argument("video_folder", help="Path to the folder containing video files.")
        parser.add_argument("audio_file", help="Path to the audio file to sync against.")
        parser.add_argument("--segment", type=float, default=300.0, help="Segment duration in seconds for analysis (default: 300).")
        parser.add_argument("--output_csv", type=str, help="Save results to a CSV file.")
        parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output.")
        parser.add_argument("--password", required=True, help="Password to run the script.")

        args = parser.parse_args()

        if args.password != "askvolx":
            console.print("[bold red]ERROR: Incorrect password. Access denied.[/bold red]")
            exit(1)

        if not os.path.isdir(args.video_folder):
            console.print(f"[red]Error: Video input '{args.video_folder}' must be a folder.[/red]")
            exit(1)

        if not os.path.isfile(args.audio_file):
            console.print(f"[red]Error: Audio input '{args.audio_file}' must be a file.[/red]")
            exit(1)

        exts = ("*.mp4", "*.mkv", "*.webm", "*.avi", "*.mov")
        video_files = [f for ext in exts for f in glob.glob(os.path.join(args.video_folder, ext))]

        if not video_files:
            console.print(f"[yellow]No compatible video files found in '{args.video_folder}'.[/yellow]")
            exit(1)

        console.print(f"Found {len(video_files)} video files. Processing against audio: {os.path.basename(args.audio_file)}")
        
        results = []
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Processing videos", total=len(video_files))
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_pair, vid_path, args.audio_file, args.segment, args.verbose) for vid_path in video_files]
                for f in as_completed(futures):
                    results.append(f.result())
                    progress.update(task, advance=1)

        display_results(results)
        if args.output_csv:
            save_results_to_csv(results, args.output_csv)

        console.print("[bold cyan]--- Script Finished ---[/bold cyan]")
    except Exception as e:
        console.print(f"[red]An unexpected error occurred: {e}[/red]")
        import traceback
        traceback.print_exc()

def display_results(results: List[Tuple[str, str, Optional[float], Optional[float], Optional[str]]]):
    """Displays results in a table."""
    table = Table(title="Audio Sync Results", show_header=True, header_style="bold magenta")
    table.add_column("Video File", style="dim", width=35)
    table.add_column("Audio File", style="dim", width=35)
    table.add_column("Start Delay (ms)", justify="right")
    table.add_column("End Delay (ms)", justify="right")
    table.add_column("Confidence", justify="center")
    table.add_column("Status", justify="left")

    for video_path, audio_path, start_delay, end_delay, err in sorted(results, key=lambda x: x[0]):
        v_name = os.path.basename(video_path)
        a_name = os.path.basename(audio_path)
        if err:
            table.add_row(v_name, a_name, "-", "-", "-", f"[red]ERROR: {err}[/red]")
        elif start_delay is not None:
            start_delay_str = f"{start_delay:+.1f}"
            end_delay_str = f"{end_delay:+.1f}" if end_delay is not None else "N/A"
            confidence_str = "-"
            if start_delay is not None and end_delay is not None:
                diff = abs(start_delay - end_delay)
                if diff < 50:
                    confidence_str = "[green]High[/green]"
                elif diff < 500:
                    confidence_str = "[yellow]Medium[/yellow]"
                else:
                    confidence_str = "[red]Low[/red]"
            table.add_row(v_name, a_name, start_delay_str, end_delay_str, confidence_str, "[green]OK[/green]")
        else:
            table.add_row(v_name, a_name, "-", "-", "-", "[red]Failed[/red]")

    console.print(table)

def save_results_to_csv(results: List[Tuple[str, str, Optional[float], Optional[float], Optional[str]]], output_csv: str):
    """Saves the results to a CSV file."""
    try:
        with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Video File", "Audio File", "Start Delay (ms)", "End Delay (ms)", "Error"])
            for video_path, audio_path, start_delay, end_delay, err in results:
                writer.writerow([
                    os.path.basename(video_path),
                    os.path.basename(audio_path),
                    start_delay if start_delay is not None else "",
                    end_delay if end_delay is not None else "",
                    err if err is not None else ""
                ])
        console.print(f"[green]Results successfully saved to {output_csv}[/green]")
    except Exception as e:
        console.print(f"[red]Error saving results to CSV: {e}[/red]")

if __name__ == "__main__":
    main()