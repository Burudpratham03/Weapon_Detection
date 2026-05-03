import argparse
import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live laptop camera object detection with YOLO11."
    )
    parser.add_argument(
        "--model",
        default="runs/weapon_yolo11n_e50/weights/best.pt",
        help="Path to trained YOLO model (.pt or .onnx).",
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=416,
                        help="Inference image size.")
    parser.add_argument(
        "--device",
        default="0",
        help="Inference device: 0 for GPU, cpu for CPU.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Run without opening a display window.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save annotated webcam video.",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/live_camera",
        help="Directory to save annotated video if --save is enabled.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Auto-stop after N frames (0 = run until q).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    camera_fps = cap.get(cv2.CAP_PROP_FPS)
    if camera_fps <= 0:
        camera_fps = 30.0

    writer = None
    output_video = None
    if args.save:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_video = output_dir / f"live_detect_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_video), fourcc, camera_fps, (width, height))

    frame_count = 0
    tick_freq = cv2.getTickFrequency()
    prev_tick = cv2.getTickCount()
    avg_fps = 0.0

    print("Starting webcam detection...")
    print("Press 'q' to stop.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame read failed. Stopping.")
                break

            results = model.predict(
                source=frame,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )

            annotated = results[0].plot()

            current_tick = cv2.getTickCount()
            dt = (current_tick - prev_tick) / tick_freq
            prev_tick = current_tick
            if dt > 0:
                instant_fps = 1.0 / dt
                if avg_fps == 0.0:
                    avg_fps = instant_fps
                else:
                    avg_fps = (0.9 * avg_fps) + (0.1 * instant_fps)

            cv2.putText(
                annotated,
                f"FPS: {avg_fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if writer is not None:
                writer.write(annotated)

            if not args.no_show:
                cv2.imshow("YOLO11 Live Detection", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("Stop requested by user.")
                    break

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed frames: {frame_count}, FPS: {avg_fps:.1f}")

            if args.max_frames > 0 and frame_count >= args.max_frames:
                print(f"Reached max frames: {args.max_frames}")
                break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    if output_video is not None:
        print(f"Saved output video: {output_video}")

    print(f"Total frames processed: {frame_count}")


if __name__ == "__main__":
    main()
