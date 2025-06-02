#!/usr/bin/env python3


import json
import subprocess
import time
import os
from pathlib import Path

def create_test_frames(frames_dir, test_chunks):
    

    import qrcode

    frames_dir.mkdir(parents=True, exist_ok=True)

    for i, chunk in enumerate(test_chunks):

        chunk_data = {
            "id": i,
            "text": chunk,
            "frame": i
        }


        qr = qrcode.QRCode(version=1, box_size=10, border=4)
        qr.add_data(json.dumps(chunk_data))
        qr.make(fit=True)


        qr_img = qr.make_image(fill_color="black", back_color="white")
        frame_path = frames_dir / f"frame_{i:06d}.png"
        qr_img.save(frame_path)

    print(f"✅ Created {len(test_chunks)} test frames in {frames_dir}")

def test_ffmpeg_executor():
    

    print("🧪 Testing FFmpeg executor with H.265 encoding...")

    test_chunks = [
        "Hello, this is test chunk 1 for Docker encoding",
        "This is test chunk 2 with more content to verify H.265 compression",
        "Final test chunk with special chars: áéíóú 中文 🎥"
    ]


    work_dir = Path("/data/temp")
    work_dir.mkdir(parents=True, exist_ok=True)

    frames_dir = work_dir / "test_frames"
    output_file = work_dir / "test_output.mkv"


    if output_file.exists():
        output_file.unlink()

    try:

        create_test_frames(frames_dir, test_chunks)


        frame_files = list(frames_dir.glob("frame_*.png"))
        if len(frame_files) != len(test_chunks):
            print(f"❌ Expected {len(test_chunks)} frames, found {len(frame_files)}")
            return False

        print(f"✅ Verified {len(frame_files)} frame files exist")


        cmd = [
            'ffmpeg', '-y',
            '-framerate', '30',
            '-i', str(frames_dir / 'frame_%06d.png'),
            '-c:v', 'libx265',
            '-preset', 'medium',
            '-crf', '24',
            '-pix_fmt', 'yuv420p',
            '-x265-params', 'keyint=1:tune=stillimage',
            str(output_file)
        ]


        cmd_data = {
            "command": cmd,
            "working_dir": str(work_dir)
        }

        print(f"🎬 Running FFmpeg command...")
        print(f"   Input: {frames_dir}")
        print(f"   Output: {output_file}")


        start_time = time.time()

        executor_cmd = [
            'python3', '/scripts/ffmpeg_executor.py',
            json.dumps(cmd_data)
        ]

        result = subprocess.run(executor_cmd,
                                capture_output=True, text=True, timeout=120)

        encoding_time = time.time() - start_time

        print(f"📊 FFmpeg executor results:")
        print(f"   Return code: {result.returncode}")
        print(f"   Encoding time: {encoding_time:.1f}s")

        if result.stdout:
            print(f"   STDOUT: {result.stdout}")
        if result.stderr:
            print(f"   STDERR: {result.stderr}")

        if result.returncode == 0 and output_file.exists():
            file_size = output_file.stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            chunks_per_mb = len(test_chunks) / file_size_mb if file_size_mb > 0 else 0

            print(f"✅ FFmpeg executor test passed!")
            print(f"   Created: {file_size} bytes ({file_size_mb:.2f} MB)")
            print(f"   Encoded: {len(test_chunks)} chunks")
            print(f"   Density: {chunks_per_mb:.0f} chunks/MB")


            print("🔍 Verifying output file...")
            probe_cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json',
                         '-show_format', '-show_streams', str(output_file)]
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)

            if probe_result.returncode == 0:
                try:
                    probe_data = json.loads(probe_result.stdout)
                    video_stream = next((s for s in probe_data['streams'] if s['codec_type'] == 'video'), None)
                    if video_stream:
                        print(f"   ✅ Video codec: {video_stream.get('codec_name', 'unknown')}")
                        print(f"   ✅ Frames: {video_stream.get('nb_frames', 'unknown')}")
                        print(f"   ✅ Resolution: {video_stream.get('width')}x{video_stream.get('height')}")
                    else:
                        print(f"   ⚠️  No video stream found in probe data")
                except json.JSONDecodeError:
                    print(f"   ⚠️  Could not parse ffprobe output")
            else:
                print(f"   ⚠️  ffprobe failed: {probe_result.stderr}")

            return True
        else:
            print(f"❌ FFmpeg executor test failed!")
            print(f"   Output file exists: {output_file.exists()}")
            if output_file.exists():
                print(f"   Output file size: {output_file.stat().st_size}")
            return False

    except subprocess.TimeoutExpired:
        print("❌ FFmpeg executor test timed out")
        return False
    except Exception as e:
        print(f"❌ FFmpeg executor test error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:

        try:
            if frames_dir.exists():
                import shutil
                shutil.rmtree(frames_dir)
            if output_file.exists():
                output_file.unlink()
        except Exception as e:
            print(f"⚠️  Cleanup error: {e}")

def test_basic_functionality():
    

    print("🧪 Testing basic container functionality...")

    tests = [
        (['python3', '--version'], "Python"),
        (['ffmpeg', '-version'], "FFmpeg"),
        (['python3', '-c', 'import json; print("JSON OK")'], "JSON import"),
        (['python3', '-c', 'import qrcode; print("QRCode OK")'], "QRCode import")
    ]

    for cmd, name in tests:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"   ✅ {name}: OK")
            else:
                print(f"   ❌ {name}: Failed (code {result.returncode})")
                if result.stderr:
                    print(f"      Error: {result.stderr}")
                return False
        except Exception as e:
            print(f"   ❌ {name}: Error - {e}")
            return False


    try:
        os.makedirs("/data/temp", exist_ok=True)
        test_file = Path("/data/temp/test.txt")
        test_file.write_text("test")
        if test_file.read_text() == "test":
            print(f"   ✅ File system: OK")
            test_file.unlink()
        else:
            print(f"   ❌ File system: Read/write failed")
            return False
    except Exception as e:
        print(f"   ❌ File system: Error - {e}")
        return False

    return True

def main():
    print("🎬 Docker Encoding Pipeline Test")
    print("=" * 50)


    if not test_basic_functionality():
        print("\n❌ Basic functionality tests failed")
        return False

    print("\n" + "=" * 50)


    if not test_ffmpeg_executor():
        print("\n❌ FFmpeg encoding test failed")
        return False

    print("\n🎉 All Docker encoding tests passed!")
    print("\n💡 Ready to use:")
    print("   encoder = FramerecallEncoder()")
    print("   encoder.build_video('output.mkv', 'index.json', codec='h265')")

    return True

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)