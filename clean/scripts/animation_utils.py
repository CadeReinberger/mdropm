"""Shared trajectory loading and GIF/MP4 writing for the animation scripts."""
from contextlib import contextmanager, ExitStack
from pathlib import Path
import pickle

from matplotlib.animation import FFMpegWriter, PillowWriter, writers
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "archived/flat_heightfield/results.pkl"


def load_results(path):
    with Path(path).open("rb") as stream:
        result = pickle.load(stream)
    times = np.asarray(result["out_t"], dtype=float)
    states = np.asarray(result["out_x"], dtype=float)
    if (times.ndim != 1 or states.ndim != 2 or len(times) < 2
            or len(times) != len(states) or states.shape[1] < 9
            or states.shape[1] % 3 or not np.all(np.diff(times) > 0)
            or not np.all(np.isfinite(times)) or not np.all(np.isfinite(states))):
        raise ValueError(f"Invalid trajectory in {path}")
    return times, states


def interpolate_states(times, states, frame_times):
    return np.column_stack([np.interp(frame_times, times, column) for column in states.T])


def add_animation_arguments(parser, stem):
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/animations")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--format", choices=("both", "gif", "mp4"), default="both")
    parser.set_defaults(output_stem=stem)


def validate_animation_arguments(parser, args):
    if min(args.frames, args.fps, args.dpi) <= 0:
        parser.error("frames, fps, and dpi must be positive")
    if args.format in {"both", "mp4"} and not writers.is_available("ffmpeg"):
        parser.error("MP4 output needs ffmpeg on PATH; install it or use --format gif")


@contextmanager
def animation_writers(figure, args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    active = []
    with ExitStack() as stack:
        for extension in ("gif", "mp4"):
            if args.format not in {"both", extension}:
                continue
            path = args.output_dir / f"{args.output_stem}.{extension}"
            writer = (PillowWriter(fps=args.fps) if extension == "gif" else
                      FFMpegWriter(fps=args.fps, codec="libx264", extra_args=[
                          "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]))
            stack.enter_context(writer.saving(figure, str(path), args.dpi))
            active.append(writer)
        yield active
    print(f"Saved {args.output_stem} animation ({args.format}) to {args.output_dir.resolve()}")
