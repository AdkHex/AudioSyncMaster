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
import re
import csv
try:
    from pymediainfo import MediaInfo
except ImportError:
    MediaInfo = None # type: ignore
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, SpinnerColumn
from rich.prompt import Prompt

# Initialize Rich Console
console = Console()


def load_audio(path: str, sr: int, duration: Optional[float] = None, offset: float = 0, verbose: bool = False) -> Optional[np.ndarray]:
    """
    Loads audio from a file, handling video extraction via in-memory pipe.
    Can load a segment from a specific offset.
    Returns a NumPy array or None if loading fails.
    """
    video_exts = ['.mp4', '.mkv', '.webm', '.avi', '.mov', '.eac3', '.ac3']
    _, ext = os.path.splitext(path)
    is_video = ext.lower() in video_exts

    if not is_video:
        try:
            # Use soundfile for more format support and precision
            with sf.SoundFile(path, 'r') as f:
                # Seek to offset
                seek_frame = int(offset * f.samplerate)
                f.seek(seek_frame)
                # Read duration
                read_frames = -1 if duration is None else int(duration * f.samplerate)
                y = f.read(frames=read_frames, dtype='float32', always_2d=True)
                # Convert to mono
                y = librosa.to_mono(y.T)
                # Resample if necessary
                if f.samplerate != sr:
                    y = librosa.resample(y, orig_sr=f.samplerate, target_sr=sr)
                return y
        except Exception:
            # If soundfile fails, fallback to librosa before trying ffmpeg
            try:
                y, native_sr = librosa.load(path, sr=None, mono=True, duration=duration, offset=offset)
                if native_sr != sr:
                    y = librosa.resample(y, orig_sr=native_sr, target_sr=sr)
                return y
            except Exception as e_librosa:
                console.print(f"[yellow]Warning: Direct load and librosa failed for {os.path.basename(path)}: {e_librosa}. Trying FFmpeg.[/yellow]")
                # Fall through to FFmpeg extraction

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
            # Fallback for containers with audio but no video stream (e.g. e-ac3 in mkv)
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
    """
    Gets the duration of an audio or video file in seconds using a fallback mechanism.
    1. MediaInfo (more robust for tricky formats)
    2. ffprobe (fast)
    3. librosa (audio-only fallback)
    """
    # 1. Try MediaInfo first as it's often more robust for video containers
    if MediaInfo:
        try:
            media_info = MediaInfo.parse(path)
            # Find the video or audio track to get duration
            duration_s = None
            for track in media_info.tracks:
                if track.duration:
                    duration_s = float(track.duration) / 1000.0
                    if verbose: console.print(f"[dim]get_duration (mediainfo): {duration_s}s for {os.path.basename(path)}[/dim]")
                    return duration_s
        except Exception as e:
            if verbose: console.print(f"[dim]pymediainfo failed for {os.path.basename(path)}: {e}[/dim]")
    else:
        if verbose: console.print("[dim]pymediainfo library not installed, skipping.[/dim]")

    # 2. Try ffprobe
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


    # 3. Try librosa as a last resort
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
    """
    Estimate delay (in ms) using fast FFT cross-correlation.
    Returns: delay_ms (float). Positive if secondary is delayed relative to primary.
    """
    def _normalize(y: np.ndarray) -> np.ndarray:
        y -= np.mean(y)
        std = np.std(y)
        if std > 1e-8:
            y /= std
        return y

    y_p = _normalize(primary_audio)
    y_s = _normalize(secondary_audio)

    # Cross-correlate to find the lag
    corr = fftconvolve(y_p, y_s[::-1], mode='full')
    # The lag is the offset of the peak from the center
    lag = np.argmax(corr) - (len(y_s) - 1)
    delay_sec = lag / sr
    return delay_sec * 1000


def process_pair(
    primary_path: str, secondary_path: str, segment_sec: float, verbose: bool = False,
    progress_callback: Optional[Callable[[int], None]] = None
) -> Tuple[str, str, Optional[float], Optional[float], Optional[str]]:
    """
    Processes a single pair of files from both start and end,
    and returns the delays.
    """
    fast_sr = 8000
    start_delay: Optional[float] = None
    end_delay: Optional[float] = None

    try:
        # --- START ANALYSIS ---
        primary_audio_start = load_audio(primary_path, sr=fast_sr, duration=segment_sec, verbose=verbose)
        if primary_audio_start is None:
            return primary_path, secondary_path, None, None, f"Failed to load start of primary: {os.path.basename(primary_path)}"

        secondary_audio_start = load_audio(secondary_path, sr=fast_sr, duration=segment_sec, verbose=verbose)
        if secondary_audio_start is None:
            return primary_path, secondary_path, None, None, f"Failed to load start of secondary: {os.path.basename(secondary_path)}"

        min_len_start = min(len(primary_audio_start), len(secondary_audio_start))
        if min_len_start > fast_sr:  # Ensure at least 1s of audio
            start_delay = estimate_sync_offset_crosscorr(
                primary_audio_start[:min_len_start], secondary_audio_start[:min_len_start], sr=fast_sr
            )
            if progress_callback:
                progress_callback(50)
        else:
            return primary_path, secondary_path, None, None, "Insufficient audio at start for analysis."

        # --- END ANALYSIS ---
        primary_duration = get_audio_duration(primary_path, verbose=verbose)
        secondary_duration = get_audio_duration(secondary_path, verbose=verbose)

        if primary_duration is None or secondary_duration is None:
            return primary_path, secondary_path, start_delay, None, "Could not get duration for end analysis."

        # Load audio from the end, ensuring we don't request a negative offset
        primary_offset = max(0, primary_duration - segment_sec)
        secondary_offset = max(0, secondary_duration - segment_sec)

        primary_audio_end = load_audio(primary_path, sr=fast_sr, duration=segment_sec, offset=primary_offset, verbose=verbose)
        if primary_audio_end is None:
            return primary_path, secondary_path, start_delay, None, f"Failed to load end of primary: {os.path.basename(primary_path)}"

        secondary_audio_end = load_audio(secondary_path, sr=fast_sr, duration=segment_sec, offset=secondary_offset, verbose=verbose)
        if secondary_audio_end is None:
            return primary_path, secondary_path, start_delay, None, f"Failed to load end of secondary: {os.path.basename(secondary_path)}"

        min_len_end = min(len(primary_audio_end), len(secondary_audio_end))

        if min_len_end > fast_sr:  # Ensure at least 1s of audio
            # The delay from the end needs to be adjusted by the difference in durations
            end_delay_raw = estimate_sync_offset_crosscorr(
                primary_audio_end[:min_len_end], secondary_audio_end[:min_len_end], sr=fast_sr
            )
            # Duration difference in ms
            duration_diff_ms = (primary_duration - secondary_duration) * 1000
            # The delay calculated at the end (`end_delay_raw`) is a combination of the true
            # offset and an artificial shift caused by the difference in file durations.
            # To get the true offset, we must add the duration difference to the raw end delay.
            # Let's trace: offset = true delay we want.
            # end_delay_raw_sec = (D_s - D_p) + offset_sec
            # offset_sec = end_delay_raw_sec - (D_s - D_p) = end_delay_raw_sec + (D_p - D_s)
            # offset_ms = end_delay_raw_ms + duration_diff_ms
            end_delay = end_delay_raw + duration_diff_ms

        else:
            return primary_path, secondary_path, start_delay, None, "Insufficient audio at end for analysis."

        return (primary_path, secondary_path, start_delay, end_delay, None)

    except Exception as e:
        # Return any successfully calculated delay along with the error
        return (primary_path, secondary_path, start_delay, end_delay, str(e))

def find_matching_files(primary_folder: str, secondary_folder: str, custom_pattern: Optional[str], verbose: bool = False) -> List[Tuple[str, str]]:
    """Matches files in two folders based on season/episode numbers or other patterns."""
    console.print(f"Searching for primary files in: [cyan]{primary_folder}[/cyan]")
    try:
        primary_files_list = os.listdir(primary_folder)
        if not primary_files_list:
            console.print(f"[yellow]Warning: No files found in the primary folder: {primary_folder}[/yellow]")
            return []
    except FileNotFoundError:
        console.print(f"[red]Error: Primary folder not found: {primary_folder}[/red]")
        return []
    primary_files = {f: os.path.join(primary_folder, f) for f in primary_files_list}
    if verbose:
        console.print(f"[dim]Found {len(primary_files)} primary files: {list(primary_files.keys())}[/dim]")

    console.print(f"Searching for secondary files in: [cyan]{secondary_folder}[/cyan]")
    try:
        secondary_files_list = os.listdir(secondary_folder)
        if not secondary_files_list:
            console.print(f"[yellow]Warning: No files found in the secondary folder: {secondary_folder}[/yellow]")
            return []
    except FileNotFoundError:
        console.print(f"[red]Error: Secondary folder not found: {secondary_folder}[/red]")
        return []
    secondary_files = {f: os.path.join(secondary_folder, f) for f in secondary_files_list}

    if verbose:
        console.print(f"[dim]Found {len(primary_files)} primary files: {list(primary_files.keys())}[/dim]")
        console.print(f"[dim]Found {len(secondary_files)} secondary files: {list(secondary_files.keys())}[/dim]")

    def get_match_key(filename, pattern):
        match = pattern.search(filename)
        return match.groups() if match else None

    patterns_to_try = []
    if custom_pattern:
        patterns_to_try.append(re.compile(custom_pattern))
    else:
        patterns_to_try.extend([
            re.compile(r'[Ss](\d+)[Ee](\d+)'),      # S01E01
            re.compile(r'(\d+)x(\d+)'),              # 1x01
            re.compile(r'[._\s-](\d{1,3})[._\s-]'), # .01.
        ])

    for pattern in patterns_to_try:
        if verbose:
            console.print(f"[dim]Attempting to match with pattern: {pattern.pattern}[/dim]")

        primary_map = {get_match_key(name, pattern): path for name, path in primary_files.items()}
        secondary_map = {get_match_key(name, pattern): path for name, path in secondary_files.items()}
        # Filter out None keys which indicate no match
        primary_map.pop(None, None)
        secondary_map.pop(None, None)

        if verbose:
            console.print(f"[dim]Primary keys: {list(primary_map.keys())}[/dim]")
            console.print(f"[dim]Secondary keys: {list(secondary_map.keys())}[/dim]")

        if primary_map and secondary_map:
            common_keys = set(primary_map.keys()) & set(secondary_map.keys())
            if common_keys:
                console.print(f"[green]Successfully matched {len(common_keys)} file(s) using pattern: {pattern.pattern}[/green]")
                break
    else:
        console.print("[yellow]Could not find matches with standard patterns, falling back to any numbers.[/yellow]")
        pattern = re.compile(r'\d+') # FIX: Changed from \d+ to 
        primary_map = {tuple(re.findall(pattern, name)): path for name, path in primary_files.items() if re.search(pattern, name)}
        secondary_map = {tuple(re.findall(pattern, name)): path for name, path in secondary_files.items() if re.search(pattern, name)}
        if verbose:
            console.print(f"Attempting fallback pattern: [bold yellow]'{pattern.pattern}'[/bold yellow]")
            console.print(f"  - Primary matches found: {len(primary_map)}")
            console.print(f"  - Secondary matches found: {len(secondary_map)}")

    matched_pairs = []
    for key, p_path in primary_map.items():
        if key in secondary_map:
            matched_pairs.append((p_path, secondary_map[key]))

    if not matched_pairs and (primary_map or secondary_map):
        console.print("[yellow]Warning: Match patterns found files, but no file pairs had a common key.[/yellow]")
        if verbose:
            console.print(f"Primary keys: {list(primary_map.keys())}")
            console.print(f"Secondary keys: {list(secondary_map.keys())}")

    return sorted(matched_pairs)


def main():
    """Main function to run the audio delay script."""
    try:
        console.print("[bold green]Script execution started.[/bold green]")
        parser = argparse.ArgumentParser(
            description="Audio delay synchronization script by volx.",
            formatter_class=argparse.RawTextHelpFormatter
        )

        # --- Arguments ---
        parser.add_argument("primary", help="Path to the primary video file or folder.")
        parser.add_argument("secondary", help="Path to the secondary audio file or folder.")

        # --- Modes ---
        mode_group = parser.add_argument_group('Processing Modes')
        mode = mode_group.add_mutually_exclusive_group(required=True)
        mode.add_argument("--single", action='store_true', help="Process a single primary file against a single secondary file.")
        mode.add_argument("--batch", action='store_true', help="Process a folder of primary videos against a single secondary audio file.")
        mode.add_argument("--series", action='store_true', help="Process a folder of primary videos against a folder of secondary audios, matching by name.")

        # --- Options ---
        options = parser.add_argument_group('Processing Options')
        options.add_argument("--crosscorr_segment", type=float, default=300.0, help="Initial segment duration in seconds for analysis (default: 300).")
        options.add_argument("--match_pattern", type=str, help="Custom regex for matching files in series mode.")
        options.add_argument("--output_csv", type=str, help="Save results to a CSV file.")
        options.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output for debugging.")

        # --- Security ---
        security = parser.add_argument_group('Security')
        security.add_argument("--password", required=True, help="Password to run the script.")

        args = parser.parse_args()
        console.print("[dim]Arguments parsed successfully.[/dim]")

        if args.password != "askvolx":
            console.print("[bold red]ERROR: Incorrect password. Access denied.[/bold red]")
            exit(1)
        
        console.print("[dim]Password check passed.[/dim]")

        console.print("[bold cyan]--- Audio Synchronization Script ---[/bold cyan]")

        results: List[Tuple[str, str, Optional[float], Optional[float], Optional[str]]] = []
        if args.series:
            results = process_series(args)
        elif args.batch:
            results = process_batch(args)
        elif args.single:
            results = process_single(args)

        if results:
            display_results(results)
            if args.output_csv:
                save_results_to_csv(results, args.output_csv)

        console.print("[bold cyan]--- Script Finished ---[/bold cyan]")
    except Exception as e:
        print(f"AN UNEXPECTED ERROR OCCURRED: {e}")
        import traceback
        traceback.print_exc()


def process_single(args: argparse.Namespace) -> List[Tuple[str, str, Optional[float], Optional[float], Optional[str]]]:
    """Processes a single pair of files."""
    console.print(f"Primary: [cyan]{os.path.basename(args.primary)}[/cyan]")
    console.print(f"Secondary: [cyan]{os.path.basename(args.secondary)}[/cyan]")

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        progress.add_task("Calculating delay...", total=None)
        result = process_pair(args.primary, args.secondary, args.crosscorr_segment, args.verbose)

    return [result]


def process_batch(args: argparse.Namespace) -> List[Tuple[str, str, Optional[float], Optional[float], Optional[str]]]:
    """Processes a folder of primary videos against a single secondary audio file."""
    if not os.path.isdir(args.primary):
        console.print(f"[red]Error: Primary input '{args.primary}' must be a folder for batch mode.[/red]")
        return []
    if not os.path.isfile(args.secondary):
        console.print(f"[red]Error: Secondary input '{args.secondary}' must be a file for batch mode.[/red]")
        return []

    exts = ("*.wav", "*.mp3", "*.aac", "*.flac", "*.ogg", "*.m4a", "*.eac3", "*.ac3", "*.mp4", "*.mkv", "*.webm", "*.avi", "*.mov")
    primary_files = [f for ext in exts for f in glob.glob(os.path.join(args.primary, ext))]

    if not primary_files:
        console.print(f"[yellow]No compatible files found in '{args.primary}'.[/yellow]")
        return []

    console.print(f"Found {len(primary_files)} primary files. Processing...")
    results = []
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Batch processing", total=len(primary_files))
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_pair, pri_path, args.secondary, args.crosscorr_segment, args.verbose) for pri_path in primary_files]
            for f in as_completed(futures):
                results.append(f.result())
                progress.update(task, advance=1)

    return results


def process_series(args: argparse.Namespace) -> List[Tuple[str, str, Optional[float], Optional[float], Optional[str]]]:
    """Processes two folders by matching file names."""
    console.print("[dim]Entered process_series function.[/dim]")
    if not os.path.isdir(args.primary) or not os.path.isdir(args.secondary):
        console.print("[red]Error: Both primary and secondary inputs must be folders for series mode.[/red]")
        return []

    matched_pairs = find_matching_files(args.primary, args.secondary, args.match_pattern, args.verbose)
    if not matched_pairs:
        console.print("[yellow]No matching file pairs found.[/yellow]")
        return []

    console.print(f"Found {len(matched_pairs)} matching pairs. Processing...")
    results = []
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Series processing", total=len(matched_pairs))
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(process_pair, p, s, args.crosscorr_segment, args.verbose): s for p, s in matched_pairs}
            for f in as_completed(futures):
                results.append(f.result())
                progress.update(task, advance=1)

    return results


def display_results(results: List[Tuple[str, str, Optional[float], Optional[float], Optional[str]]]):
    """Displays batch or series results in a table."""
    table = Table(title="Processing Results", show_header=True, header_style="bold magenta")
    table.add_column("Primary File", style="dim", width=35)
    table.add_column("Secondary File", style="dim", width=35)
    table.add_column("Start Delay (ms)", justify="right")
    table.add_column("End Delay (ms)", justify="right")
    table.add_column("Confidence", justify="center")
    table.add_column("Status", justify="left")

    for primary_path, secondary_path, start_delay, end_delay, err in sorted(results, key=lambda x: x[0]):
        p_name = os.path.basename(primary_path)
        s_name = os.path.basename(secondary_path)
        if err:
            table.add_row(p_name, s_name, "-", "-", "-", f"[red]ERROR: {err}[/red]")
        elif start_delay is not None:
            start_delay_str = f"{start_delay:+.1f}"
            end_delay_str = f"{end_delay:+.1f}" if end_delay is not None else "N/A"
            confidence_str = "-"
            if start_delay is not None and end_delay is not None:
                diff = abs(start_delay - end_delay)
                if diff < 50: # Threshold for high confidence
                    confidence_str = "[green]High[/green]"
                elif diff < 500: # Threshold for medium confidence
                    confidence_str = "[yellow]Medium[/yellow]"
                else:
                    confidence_str = "[red]Low[/red]"
            table.add_row(p_name, s_name, start_delay_str, end_delay_str, confidence_str, "[green]OK[/green]")
        else:
            table.add_row(p_name, s_name, "-", "-", "-", "[red]Failed[/red]")

    console.print(table)


def save_results_to_csv(results: List[Tuple[str, str, Optional[float], Optional[float], Optional[str]]], output_csv: str):
    """Saves the processing results to a CSV file."""
    try:
        with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Primary File", "Secondary File", "Start Delay (ms)", "End Delay (ms)", "Error"])
            for primary_path, secondary_path, start_delay, end_delay, err in results:
                writer.writerow([
                    os.path.basename(primary_path),
                    os.path.basename(secondary_path),
                    start_delay if start_delay is not None else "",
                    end_delay if end_delay is not None else "",
                    err if err is not None else ""
                ])
        console.print(f"[green]Results successfully saved to {output_csv}[/green]")
    except Exception as e:
        console.print(f"[red]Error saving results to CSV: {e}[/red]")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        console.print(f"[red]An unexpected error occurred in the main execution: {e}[/red]")
        import traceback
        traceback.print_exc()
