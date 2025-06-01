#!/usr/bin/env python3


import sys
import argparse
import time
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from framerecall.encoder import FramerecallEncoder
except ImportError:
    print("❌ Could not import FramerecallEncoder. Make sure you're running from the project root.")
    sys.exit(1)

def format_size(bytes_size):
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"

def find_files_in_directory(dir_path, file_types=None, max_files=None):
    

    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        return []


    if file_types is None:
        file_types = ['pdf', 'epub', 'txt', 'md', 'json']

    files = []

    print(f"🔍 Scanning directory: {dir_path}")
    print(f"   Looking for: {', '.join(file_types)} files")

    for file_type in file_types:
        pattern = f"**/*.{file_type}"
        found = list(dir_path.glob(pattern))
        files.extend(found)
        if found:
            print(f"   Found {len(found)} .{file_type} files")


    files = sorted(list(set(files)))

    if max_files and len(files) > max_files:
        print(f"⚠️  Found {len(files)} files, limiting to first {max_files}")
        files = files[:max_files]

    print(f"📁 Total files to process: {len(files)}")
    return files

def load_multiple_files(file_paths, chunk_size=512, show_progress=True):
    

    encoder = FramerecallEncoder()
    file_stats = []

    if show_progress:
        from tqdm import tqdm
        file_iter = tqdm(file_paths, desc="Loading files")
    else:
        file_iter = file_paths

    total_files_processed = 0
    total_files_failed = 0

    for file_path in file_iter:
        try:
            file_encoder = FramerecallEncoder()

            if file_path.suffix.lower() == '.pdf':
                file_encoder.add_pdf(str(file_path), chunk_size=chunk_size)
            elif file_path.suffix.lower() == '.epub':
                file_encoder.add_epub(str(file_path), chunk_size=chunk_size)
            elif file_path.suffix.lower() == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                if isinstance(chunks, list):
                    file_encoder.add_chunks(chunks)
            else:

                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                file_encoder.add_text(text, chunk_size=chunk_size)

            if file_encoder.chunks:

                encoder.chunks.extend(file_encoder.chunks)

                file_stats.append({
                    'name': file_path.name,
                    'path': str(file_path),
                    'chunks': len(file_encoder.chunks),
                    'size_bytes': file_path.stat().st_size,
                    'characters': sum(len(chunk) for chunk in file_encoder.chunks)
                })
                total_files_processed += 1
            else:
                print(f"⚠️  No content extracted from {file_path.name}")
                total_files_failed += 1

        except Exception as e:
            print(f"❌ Error processing {file_path.name}: {e}")
            total_files_failed += 1

    total_chars = sum(len(chunk) for chunk in encoder.chunks)
    total_size = sum(stat['size_bytes'] for stat in file_stats)

    summary = {
        'files_processed': total_files_processed,
        'files_failed': total_files_failed,
        'total_chunks': len(encoder.chunks),
        'total_characters': total_chars,
        'total_file_size': total_size,
        'file_stats': file_stats
    }

    print(f"\n📊 Collection Summary:")
    print(f"   ✅ Files processed: {total_files_processed}")
    if total_files_failed > 0:
        print(f"   ❌ Files failed: {total_files_failed}")
    print(f"   📋 Total chunks: {len(encoder.chunks)}")
    print(f"   📄 Total content: {format_size(total_chars)} characters")
    print(f"   💾 Total file size: {format_size(total_size)}")

    return encoder, summary
def load_user_data(input_path, chunk_size=512, file_types=None, max_files=None):
    

    input_path = Path(input_path)

    if not input_path.exists():
        print(f"❌ Path not found: {input_path}")
        return None, None


    if input_path.is_dir():
        print(f"📂 Loading directory: {input_path}")


        files = find_files_in_directory(input_path, file_types, max_files)

        if not files:
            print(f"❌ No supported files found in {input_path}")
            return None, None


        encoder, summary = load_multiple_files(files, chunk_size, show_progress=True)

        if not encoder.chunks:
            print("❌ No content extracted from any files")
            return None, None


        info = {
            'type': 'directory',
            'path': str(input_path),
            'files_processed': summary['files_processed'],
            'files_failed': summary['files_failed'],
            'chunks': summary['total_chunks'],
            'total_chars': summary['total_characters'],
            'total_file_size': summary['total_file_size'],
            'avg_chunk_size': summary['total_characters'] / summary['total_chunks'] if summary['total_chunks'] > 0 else 0,
            'file_stats': summary['file_stats']
        }

        return encoder, info

    else:

        print(f"📂 Loading file: {input_path.name}")

        encoder = FramerecallEncoder()

        try:
            if input_path.suffix.lower() == '.pdf':
                print("   📄 Detected PDF file")
                encoder.add_pdf(str(input_path), chunk_size=chunk_size)

            elif input_path.suffix.lower() == '.epub':
                print("   📚 Detected EPUB file")
                encoder.add_epub(str(input_path), chunk_size=chunk_size)

            elif input_path.suffix.lower() == '.json':
                print("   📋 Detected JSON chunks file")
                with open(input_path, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)

                if isinstance(chunks, list):
                    encoder.add_chunks(chunks)
                    print(f"   ✅ Loaded {len(chunks)} pre-made chunks")
                else:
                    print("❌ JSON file must contain a list of text chunks")
                    return None, None

            else:
                print("   📝 Treating as text file")
                with open(input_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                encoder.add_text(text, chunk_size=chunk_size)

        except Exception as e:
            print(f"❌ Error loading file: {e}")
            return None, None

        if not encoder.chunks:
            print("❌ No content extracted from file")
            return None, None

        total_chars = sum(len(chunk) for chunk in encoder.chunks)
        avg_chunk_size = total_chars / len(encoder.chunks)

        print(f"   ✅ Extracted {len(encoder.chunks)} chunks")
        print(f"   📊 Total content: {format_size(total_chars)} characters")
        print(f"   📏 Average chunk size: {avg_chunk_size:.0f} characters")

        info = {
            'type': 'file',
            'file_name': input_path.name,
            'file_size': format_size(input_path.stat().st_size),
            'chunks': len(encoder.chunks),
            'total_chars': total_chars,
            'avg_chunk_size': avg_chunk_size
        }

        return encoder, info

def run_codec_comparison(encoder, data_info, output_dir="output"):
    

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    if data_info['type'] == 'directory':

        name_stem = Path(data_info['path']).name
    else:

        name_stem = Path(data_info['file_name']).stem


    timestamp = int(time.time())
    name_stem = f"{name_stem}_{timestamp}"

    print(f"\n🏁 Starting codec comparison")
    print("=" * 60)

    results = {}


    print("\n🏠 Test 1: Native MP4V Encoding")
    print("-" * 30)

    mp4v_path = output_path / f"{name_stem}_mp4v.mp4"
    mp4v_index = output_path / f"{name_stem}_mp4v.json"

    start_time = time.time()
    try:
        mp4v_stats = encoder.build_video(
            str(mp4v_path),
            str(mp4v_index),
            codec="mp4v",
            show_progress=True
        )
        mp4v_time = time.time() - start_time

        results['mp4v'] = {
            'success': True,
            'file_size': mp4v_path.stat().st_size,
            'file_size_mb': mp4v_stats.get('video_size_mb', 0),
            'encoding_time': mp4v_time,
            'backend': mp4v_stats.get('backend', 'native'),
            'path': mp4v_path
        }

        print(f"✅ MP4V encoding complete")
        print(f"   File size: {format_size(results['mp4v']['file_size'])}")
        print(f"   Encoding time: {mp4v_time:.1f} seconds")

    except Exception as e:
        print(f"❌ MP4V encoding failed: {e}")
        results['mp4v'] = {'success': False, 'error': str(e)}


    print("\n🐳 Test 2: Docker H.265 Encoding")
    print("-" * 30)


    encoder2 = FramerecallEncoder()
    encoder2.chunks = encoder.chunks.copy()

    h265_path = output_path / f"{name_stem}_h265.mp4"
    h265_index = output_path / f"{name_stem}_h265.json"

    start_time = time.time()
    try:
        h265_stats = encoder2.build_video(
            str(h265_path),
            str(h265_index),
            codec="h265",
            show_progress=True,
            auto_build_docker=True,
            allow_fallback=False
        )
        h265_time = time.time() - start_time

        results['h265'] = {
            'success': True,
            'file_size': h265_path.stat().st_size,
            'file_size_mb': h265_stats.get('video_size_mb', 0),
            'encoding_time': h265_time,
            'backend': h265_stats.get('backend', 'unknown'),
            'path': h265_path
        }

        print(f"✅ H.265 encoding complete")
        print(f"   File size: {format_size(results['h265']['file_size'])}")
        print(f"   Encoding time: {h265_time:.1f} seconds")
        print(f"   Backend used: {results['h265']['backend']}")

    except Exception as e:
        print(f"❌ H.265 encoding failed: {e}")
        print("   Docker backend is required for H.265 comparison.")
        print("   Run 'make build' to set up Docker container, then try again.")
        results['h265'] = {'success': False, 'error': str(e)}

    return results

def print_comparison_results(data_info, results):
    

    print(f"\n📊 COMPRESSION COMPARISON RESULTS")
    print("=" * 60)

    if data_info['type'] == 'directory':
        print(f"📁 Source: {data_info['path']} ({data_info['files_processed']} files)")
        print(f"📋 Content: {data_info['chunks']} chunks, {format_size(data_info['total_chars'])} characters")
        print(f"💾 Original size: {format_size(data_info['total_file_size'])}")
    else:
        print(f"📁 Source: {data_info['file_name']} ({data_info['file_size']})")
        print(f"📋 Content: {data_info['chunks']} chunks, {format_size(data_info['total_chars'])} characters")
    print()

    if results['mp4v']['success'] and results['h265']['success']:

        mp4v_size = results['mp4v']['file_size']
        h265_size = results['h265']['file_size']

        compression_ratio = mp4v_size / h265_size if h265_size > 0 else float('inf')
        space_saved = ((mp4v_size - h265_size) / mp4v_size) * 100 if mp4v_size > 0 else 0

        chunks_count = data_info['chunks']
        chunks_per_mb_mp4v = chunks_count / (mp4v_size / (1024*1024)) if mp4v_size > 0 else 0
        chunks_per_mb_h265 = chunks_count / (h265_size / (1024*1024)) if h265_size > 0 else 0

        print(f"🏠 MP4V (Native):  {format_size(mp4v_size):>10} ({chunks_per_mb_mp4v:.0f} chunks/MB)")
        print(f"🐳 H.265 (Docker): {format_size(h265_size):>10} ({chunks_per_mb_h265:.0f} chunks/MB)")
        print()
        print(f"🎯 H.265 is {compression_ratio:.1f}x smaller than MP4V")
        print(f"💾 Space saved: {space_saved:.1f}%")
        print(f"⚡ Density improvement: {chunks_per_mb_h265/chunks_per_mb_mp4v:.1f}x more chunks per MB")


        mp4v_time = results['mp4v']['encoding_time']
        h265_time = results['h265']['encoding_time']
        time_ratio = h265_time / mp4v_time if mp4v_time > 0 else float('inf')

        print()
        print(f"⏱️  Encoding Times:")
        print(f"   MP4V:  {mp4v_time:.1f} seconds")
        print(f"   H.265: {h265_time:.1f} seconds ({time_ratio:.1f}x {'slower' if time_ratio > 1 else 'faster'})")

    else:

        print("⚠️  Partial Results:")

        if results['mp4v']['success']:
            mp4v_size = results['mp4v']['file_size']
            chunks_per_mb = data_info['chunks'] / (mp4v_size / (1024*1024)) if mp4v_size > 0 else 0
            print(f"✅ MP4V: {format_size(mp4v_size)} ({chunks_per_mb:.0f} chunks/MB)")
        else:
            print(f"❌ MP4V failed: {results['mp4v']['error']}")

        if results['h265']['success']:
            h265_size = results['h265']['file_size']
            chunks_per_mb = data_info['chunks'] / (h265_size / (1024*1024)) if h265_size > 0 else 0
            backend = results['h265']['backend']
            print(f"✅ H.265: {format_size(h265_size)} ({chunks_per_mb:.0f} chunks/MB) via {backend}")
        else:
            print(f"❌ H.265 failed: {results['h265']['error']}")

    print()
    print(f"📁 Output files saved to: {Path('output').absolute()}")

def main():
    parser = argparse.ArgumentParser(
        description="Compare H.265 vs MP4V compression on YOUR data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=
    )

    parser.add_argument('input_path', help='Path to your file or directory (PDF, EPUB, TXT, JSON, or folder)')
    parser.add_argument('--chunk-size', type=int, default=512,
                        help='Chunk size for text splitting (default: 512)')
    parser.add_argument('--output-dir', default='output',
                        help='Output directory for encoded videos (default: output)')
    parser.add_argument('--file-types', nargs='+', default=['pdf', 'epub', 'txt', 'md', 'json'],
                        help='File types to process in directories (default: pdf epub txt md json)')
    parser.add_argument('--max-files', type=int,
                        help='Maximum number of files to process from directory')
    parser.add_argument('--chunks', action='store_true',
                        help='Treat input file as pre-made JSON chunks')

    args = parser.parse_args()

    print("🎥 Framerecall Codec Comparison Tool")
    print("Compare H.265 vs MP4V compression on YOUR data")
    print()


    encoder, data_info = load_user_data(args.input_path, args.chunk_size, args.file_types, args.max_files)
    if not encoder:
        sys.exit(1)


    docker_status = encoder.get_docker_status()
    print(f"\n🐳 Docker Status: {docker_status}")

    if "not found" in docker_status:
        print("\n⚠️  H.265 encoding requires Docker. Install Docker Desktop for best results.")
        print("   H.265 test will attempt auto-build or fall back to MP4V.")


    results = run_codec_comparison(encoder, data_info, args.output_dir)


    print_comparison_results(data_info, results)

    print(f"\n🎉 Comparison complete! Now you know how YOUR data compresses.")

if __name__ == '__main__':
    main()