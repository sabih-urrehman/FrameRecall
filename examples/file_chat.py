#!/usr/bin/env python3
"""
file_chat.py - Example script for testing FrameRecallChat with external files

This script allows you to:
1. Create a memory video from your own files
2. Chat with the created memory using different LLM providers
3. Store results in output/ directory to avoid contaminating the main repo

Usage:
    python file_chat.py --input-dir /path/to/documents --provider google
    python file_chat.py --files file1.txt file2.pdf --provider openai
    python file_chat.py --load-existing output/my_memory --provider google

Examples:
    # Create memory from a directory and chat with Google
    python file_chat.py --input-dir ~/Documents/research --provider google

    # Create memory from specific files and chat with OpenAI
    python file_chat.py --files report.pdf notes.txt --provider openai

    # Load existing memory and continue chatting
    python file_chat.py --load-existing output/research_memory --provider google
"""

import argparse
import os
import sys
import time
from pathlib import Path
from datetime import datetime
import json
sys.path.insert(0, str(Path(__file__).parent.parent))  # Go up TWO levels from examples/
from framerecall import FrameRecallEncoder, FrameRecallChat
from framerecall.config import get_default_config

def setup_output_dir():
    """Create output directory if it doesn't exist"""
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    return output_dir

def generate_memory_name(input_source):
    """Generate a meaningful name for the memory files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if isinstance(input_source, list):
        # Multiple files
        base_name = f"files_{len(input_source)}items"
    else:
        # Directory
        dir_name = Path(input_source).name
        base_name = f"dir_{dir_name}"
    return f"{base_name}_{timestamp}"

def collect_files_from_directory(directory_path, extensions=None):
    """Collect supported files from a directory"""
    if extensions is None:
        extensions = {'.txt', '.md', '.pdf', '.doc', '.docx', '.rtf', '.epub', '.html', '.htm'}
    directory = Path(directory_path)
    if not directory.exists():
        raise ValueError(f"Directory does not exist: {directory_path}")
    files = []
    for ext in extensions:
        files.extend(directory.rglob(f"*{ext}"))
    return [str(f) for f in files if f.is_file()]

def create_memory_with_fallback(encoder, video_path, index_path):
    try:
        build_stats = encoder.build_video(str(video_path), str(index_path))
        return build_stats
    except Exception as e:
        error_str = str(e)
        if "is_trained" in error_str or "IndexIVFFlat" in error_str or "training" in error_str.lower():
            print(f"⚠️  FAISS IVF training failed: {e}")
            print(f"🔄 Auto-switching to Flat index for compatibility...")
            # Override config to use Flat index
            original_index_type = encoder.config["index"]["type"]
            encoder.config["index"]["type"] = "Flat"
            try:
                # Recreate the index manager with Flat index
                encoder._setup_index()
                build_stats = encoder.build_video(str(video_path), str(index_path))
                print(f"✅ Successfully created memory using Flat index")
                return build_stats
            except Exception as fallback_error:
                print(f"❌ Fallback also failed: {fallback_error}")
                raise
        else:
            raise

def create_framerecall_memory(files, output_dir, memory_name, **config_overrides):
    """Create a memory video from a list of files"""
    print(f"Creating FrameRecall memory from {len(files)} files...")
    # Start timing
    start_time = time.time()
    # Get chunking config
    config = get_default_config()
    for key, value in config_overrides.items():
        if key in ['chunk_size', 'overlap']:
            config["chunking"][key] = value
        elif key == 'index_type':
            config["index"]["type"] = value
        elif key == 'codec':
            config[key] = value
    # Initialize encoder with config first (this ensures config consistency)
    encoder = FrameRecallEncoder(config)
    # Import tqdm for progress bars
    try:
        from tqdm import tqdm
        use_progress = True
    except ImportError:
        print("Note: Install tqdm for progress bars (pip install tqdm)")
        use_progress = False
    processed_count = 0
    skipped_count = 0
    file_iterator = tqdm(files, desc="Processing files") if use_progress else files
    for file_path in file_iterator:
        file_path = Path(file_path)
        if not use_progress:
            print(f"Processing: {file_path.name}")
        try:
            chunk_size = config["chunking"]["chunk_size"]
            overlap = config["chunking"]["overlap"]
            if file_path.suffix.lower() == '.pdf':
                encoder.add_pdf(str(file_path), chunk_size, overlap)
            elif file_path.suffix.lower() == '.epub':
                encoder.add_epub(str(file_path), chunk_size, overlap)
            elif file_path.suffix.lower() in ['.html', '.htm']:
                # Process HTML with BeautifulSoup
                try:
                    from bs4 import BeautifulSoup
                except ImportError:
                    print(f"Warning: BeautifulSoup not available for HTML processing. Skipping {file_path.name}")
                    skipped_count += 1
                    continue
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    soup = BeautifulSoup(f.read(), 'html.parser')
                    for script in soup(["script", "style"]):
                        script.decompose()
                    text = soup.get_text()
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split(""))
                    clean_text = ' '.join(chunk for chunk in chunks if chunk)
                    if clean_text.strip():
                        encoder.add_text(clean_text, chunk_size, overlap)
            else:
                # Read as text file
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if content.strip():
                        encoder.add_text(content, chunk_size, overlap)
            processed_count += 1
        except Exception as e:
            print(f"Warning: Could not process {file_path.name}: {e}")
            skipped_count += 1
            continue
    processing_time = time.time() - start_time
    print(f"\n📊 Processing Summary:")
    print(f"✅ Successfully processed: {processed_count} files")
    print(f"⚠️  Skipped: {skipped_count} files")
    print(f"⏱️  Processing time: {processing_time:.2f} seconds")
    # Build the video
    if processed_count == 0:
        raise ValueError("No files were successfully processed")
    # Build the video (video_ext already determined from encoder config)
    video_path = output_dir / f"{memory_name}.mp4"
    index_path = output_dir / f"{memory_name}_index.json"
    print(f"\n🎬 Building memory video: {video_path}")
    print(f"📊 Total chunks to encode: {len(encoder.chunks)}")
    encoding_start = time.time()
    build_stats = create_memory_with_fallback(encoder, video_path, index_path)
    encoding_time = time.time() - encoding_start
    total_time = time.time() - start_time
    # Enhanced statistics
    print(f"\n🎉 Memory created successfully!")
    print(f"📁 Video: {video_path}")
    print(f"📋 Index: {index_path}")
    print(f"📊 Chunks: {build_stats.get('total_chunks', 'unknown')}")
    print(f"🎞️  Frames: {build_stats.get('total_frames', 'unknown')}")
    print(f"📏 Video size: {video_path.stat().st_size / (1024 * 1024):.1f} MB")
    print(f"⏱️  Encoding time: {encoding_time:.2f} seconds")
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    if build_stats.get('video_size_mb', 0) > 0:
        # Calculate rough compression stats
        total_chars = sum(len(chunk) for chunk in encoder.chunks)
        original_size_mb = total_chars / (1024 * 1024)  # Rough estimate
        compression_ratio = original_size_mb / build_stats['video_size_mb'] if build_stats['video_size_mb'] > 0 else 0
        print(f"📦 Estimated compression ratio: {compression_ratio:.1f}x")
    # Save metadata about this memory
    metadata = {
        'created': datetime.now().isoformat(),
        'source_files': files,
        'video_path': str(video_path),
        'index_path': str(index_path),
        'config_used': config,
        'processing_stats': {
            'files_processed': processed_count,
            'files_skipped': skipped_count,
            'processing_time_seconds': processing_time,
            'encoding_time_seconds': encoding_time,
            'total_time_seconds': total_time
        },
        'build_stats': build_stats
    }
    metadata_path = output_dir / f"{memory_name}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"📄 Metadata: {metadata_path}")
    return str(video_path), str(index_path)

def load_existing_memory(memory_path):
    """Load an existing memory from the output directory"""
    memory_path = Path(memory_path)
    # Handle different input formats
    if memory_path.is_dir():
        # Directory provided, look for memory files
        video_files = []
        for ext in ['mp4', 'avi', 'mkv']:
            video_files.extend(memory_path.glob(f"*.{ext}"))
        if not video_files:
            raise ValueError(f"No video files found in {memory_path}")
        video_path = video_files[0]
        possible_index_paths = [
            video_path.with_name(video_path.stem + '_index.json'),
            video_path.with_suffix('.json'),
            video_path.with_suffix('_index.json')
        ]
        index_path = None
        for possible_path in possible_index_paths:
            if possible_path.exists():
                index_path = possible_path
                break
        if not index_path:
            raise ValueError(f"No index file found for {video_path}")
    elif memory_path.suffix in ['.mp4', '.avi', '.mkv']:
        # Video file provided
        video_path = memory_path
        index_path = memory_path.with_name(memory_path.stem + '_index.json')
    else:
        # Assume it's a base name, try to find files
        base_path = memory_path
        video_path = None
        # Try different video extensions
        for ext in ['mp4', 'avi', 'mkv']:
            candidate = base_path.with_suffix(f'.{ext}')
            if candidate.exists():
                video_path = candidate
                break
        if not video_path:
            raise ValueError(f"No video file found with base name: {memory_path}")
        index_path = base_path.with_suffix('_index.json')
    # Check if files exist
    if not video_path.exists():
        raise ValueError(f"Video file not found: {video_path}")
    if not index_path.exists():
        raise ValueError(f"Index file not found: {index_path}")
    try:
        with open(index_path, 'r') as f:
            index_data = json.load(f)
        chunk_count = len(index_data.get('metadata', []))
        print(f"✅ Index contains {chunk_count} chunks")
    except Exception as e:
        raise ValueError(f"Index file corrupted: {e}")
    # Check video file size
    video_size_mb = video_path.stat().st_size / (1024 * 1024)
    print(f"✅ Video file: {video_size_mb:.1f} MB")
    print(f"Loading existing memory:")
    print(f"📁 Video: {video_path}")
    print(f"📋 Index: {index_path}")
    return str(video_path), str(index_path)

def start_framerecall_chat(video_path, index_path, provider='google', model=None):
    """Start an interactive chat session"""
    print(f"\nInitializing chat with {provider}...")
    try:
        chat = FrameRecallChat(
            video_file=video_path,
            index_file=index_path,
            llm_provider=provider,
            llm_model=model
        )
        print("✓ FrameRecall chat ready!")
        print("\nStarting interactive session...")
        print("Commands:")
        print("- Type your questions normally")
        print("- Type 'quit' or 'exit' to end")
        print("- Type 'clear' to clear conversation history")
        print("- Type 'stats' to see session statistics")
        print("=" * 50)
        # Start interactive chat
        while True:
            try:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    # Export conversation before exiting
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    export_path = Path("output") / f"conversation_{timestamp}.json"
                    chat.export_conversation(str(export_path))
                    print(f"💾 Conversation saved to: {export_path}")
                    print("Goodbye!")
                    break
                elif user_input.lower() == 'clear':
                    chat.clear_history()
                    print("🗑️ Conversation history cleared")
                    continue
                elif user_input.lower() == 'stats':
                    stats = chat.get_stats()
                    print(f"📊 Session stats: {stats}")
                    continue
                if not user_input:
                    continue
                # Get response (always stream for better UX)
                chat.chat(user_input, stream=True)
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    except Exception as e:
        print(f"Error initializing chat: {e}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Chat with your documents using FrameRecall with enhanced configuration options",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input-dir',
        help='Directory containing documents to process'
    )
    input_group.add_argument(
        '--files',
        nargs='+',
        help='Specific files to process'
    )
    input_group.add_argument(
        '--load-existing',
        help=f'Load existing memory (provide path to video  file or directory)'
    )
    # LLM options~
    parser.add_argument(
        '--chunk-size',
        type=int,
        help='Override default chunk size (e.g., 2048, 4096)'
    )
    parser.add_argument(
        '--overlap',
        type=int,
        help='Override default chunk overlap (e.g., 16, 32, 64)'
    )
    parser.add_argument(
        '--index-type',
        choices=['Flat', 'IVF'],
        help='FAISS index type (Flat for small datasets, IVF for large datasets)'
    )
    parser.add_argument(
        '--codec',
        choices=['h264', 'h265', 'mp4v'],
        help='Video codec to use for compression'
    )
    parser.add_argument(
        '--provider',
        choices=['openai', 'google', 'anthropic'],
        default='google',
        help='LLM provider to use (default: google)'
    )
    parser.add_argument(
        '--model',
        help='Specific model to use (uses provider defaults if not specified)'
    )
    # Memory options
    parser.add_argument(
        '--memory-name',
        help='Custom name for the memory files (auto-generated if not provided)'
    )
    # Processing options
    parser.add_argument(
        '--extensions',
        nargs='+',
        default=['.txt', '.md', '.pdf', '.doc', '.docx', '.epub', '.html', '.htm'],
        help='File extensions to include when processing directories'
    )
    args = parser.parse_args()
    # Setup output directory
    output_dir = setup_output_dir()
    try:
        # Get or create memory
        if args.load_existing:
            video_path, index_path = load_existing_memory(args.load_existing)
        else:
            # Collect files
            if args.input_dir:
                files = collect_files_from_directory(args.input_dir, set(args.extensions))
                if not files:
                    print(f"No supported files found in {args.input_dir}")
                    return 1
                print(f"Found {len(files)} files to process")
                input_source = args.input_dir
            else:
                files = args.files
                for f in files:
                    if not Path(f).exists():
                        print(f"File not found: {f}")
                        return 1
                input_source = files
            # Generate memory name
            memory_name = args.memory_name or generate_memory_name(input_source)
            # Create memory
            config_overrides = {}
            if args.chunk_size:
                config_overrides['chunk_size'] = args.chunk_size
            if args.overlap:
                config_overrides['overlap'] = args.overlap
            if args.index_type:
                config_overrides['index_type'] = args.index_type
            if args.codec:
                config_overrides['codec'] = args.codec
            # Show what defaults are being used if no overrides provided
            if not config_overrides:
                default_config = get_default_config()
                print(f"📋 Using default configuration:")
                print(f"   Chunk size: {default_config['chunking']['chunk_size']}")
                print(f"   Overlap: {default_config['chunking']['overlap']}")
                print(f"   Index type: {default_config['index']['type']}")
                print(f"   Codec: {default_config.get('codec', 'h265')}")
            video_path, index_path = create_framerecall_memory(files, output_dir, memory_name, **config_overrides)
        # Start chat session
        success = start_framerecall_chat(video_path, index_path, args.provider, args.model)
        return 0 if success else 1
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())