"""Record actual Slang-rendered frames at a fixed simulation rate to H.264."""
import time

import imageio_ffmpeg
import numpy as np


def record(app):
    args = app.args
    path = args.video
    path.parent.mkdir(parents=True, exist_ok=True)
    # Finish the container before publishing the requested filename.
    temporary = path.with_name(path.stem + ".partial.mp4")
    frames = max(1, round(args.seconds * args.fps))
    writer = imageio_ffmpeg.write_frames(str(temporary), (args.width, args.height),
        fps=args.fps, codec="libx264", pix_fmt_out="yuv420p", macro_block_size=2,
        output_params=["-crf", "18", "-preset", "medium", "-movflags", "+faststart"])
    start = time.perf_counter()
    try:
        writer.send(None)
        for i in range(frames):
            encoder = app.device.create_command_encoder()
            app.advance(encoder, 1.0 / args.fps)
            app.water.render(encoder)
            app.device.submit_command_buffer(encoder.finish())
            app.device.wait()
            rgb = app.water.display.to_numpy()[..., :3]
            if not np.isfinite(rgb).all():
                raise RuntimeError("nonfinite video frame")
            writer.send(np.ascontiguousarray(np.clip(rgb * 255 + 0.5, 0, 255), dtype=np.uint8))
            if i == frames // 2:
                app.water.save(path.with_suffix(".png"))
            if (i + 1) % (args.fps * 2) == 0:
                print(f"Recording {i + 1}/{frames} frames | {time.perf_counter() - start:.1f}s elapsed", flush=True)
        writer.close()
        actual_frames, duration = imageio_ffmpeg.count_frames_and_secs(str(temporary))
        if actual_frames != frames or abs(duration - frames / args.fps) > 0.05:
            raise RuntimeError(f"Video verification failed: {actual_frames} frames, {duration}s")
        temporary.replace(path)
    finally:
        writer.close()
        if temporary.exists():
            temporary.unlink()
    print(f"Saved {path} | {actual_frames} frames | {duration:.3f}s | {args.width}x{args.height} @ {args.fps} fps", flush=True)
