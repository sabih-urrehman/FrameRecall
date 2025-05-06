#!/usr/bin/env python3
"""
Demo: Generate clip-based memory archive and search map from textual input
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framerecall import FrameRecallEncoder
import time

def main():
    # Sample input - could be fetched from APIs, logs, etc.
    segments = [
        "In early 2024, a new AI benchmark reached human-level math reasoning.",
        "Neural engines surpassed 2 trillion operations per second last quarter.",
        "Emerging chipsets offer tenfold gains for inference latency reduction.",
        "Cloud bandwidth pricing dropped 70% in three years across regions.",
        "Quantum-safe protocols are rolling out for industrial-grade security.",
        "Fog computing supports sub-millisecond responsiveness at the edge.",
        "Generative models now produce full video scenes from brief prompts.",
        "Consensus algorithms process hundreds of thousands of ops per second.",
        "Wi-Fi 7 enables gigabit connectivity in dense environments.",
        "Self-driving systems accumulated over 60M autonomous miles to date.",
        "Speech understanding now achieves near-human fidelity in 20+ tongues.",
        "Workflow bots reduce overhead costs across entire enterprise stacks.",
        "Mixed reality headsets last all day on a single charge.",
        "Face and gait recognition errors fall below 0.0005% in top systems.",
        "Volunteer computing aggregates petaflops of surplus device power.",
        "Modular server farms now operate on 100% green infrastructure.",
        "Smart agents preserve context across lengthy interactive sequences.",
        "Threat detection systems react dozens of times faster than manual review.",
        "Simulated urban twins help redesign city transport frameworks.",
        "Synthetic voice creation only needs a 2-second audio snippet now."
    ]

    print("FrameRecall Demo: Compiling Clip Archive")
    print("=" * 50)

    # Initialize encoder
    encoder = FrameRecallEncoder()

    # Insert segments
    print(f"\nInjecting {len(segments)} segments into encoder...")
    encoder.add_chunks(segments)

    # Load extended content with chunking enabled
    supplementary_text = """
    Next-gen computing will stem from layered tech integration.
    Quantum breakthroughs will crack challenges classical systems can't handle.
    Smart algorithms will permeate nearly every digital experience.
    Edge-cloud symbiosis will intelligently place workloads for efficiency.
    New privacy layers will allow group work while keeping info shielded.
    """

    print("\nLoading supplementary content with auto segmentation...")
    encoder.add_text(supplementary_text, chunk_size=100, overlap=20)

    # Display encoder summary
    stats = encoder.get_stats()
    print(f"\nEncoder Summary:")
    print(f"  Total segments: {stats['total_chunks']}")
    print(f"  Character count: {stats['total_characters']}")
    print(f"  Avg segment length: {stats['avg_chunk_size']:.1f} characters")

    # Generate outputs
    target_dir = "output"
    os.makedirs(target_dir, exist_ok=True)

    clip_path = os.path.join(target_dir, "archive.mp4")
    map_path = os.path.join(target_dir, "search_index.json")

    print(f"\nGenerating clip: {clip_path}")
    print(f"Generating index map: {map_path}")

    start_time = time.time()
    build_stats = encoder.build_video(clip_path, map_path, show_progress=True)
    elapsed = time.time() - start_time

    print(f"\nCompilation done in {elapsed:.2f} seconds")
    print(f"\nClip Details:")
    print(f"  Length: {build_stats['duration_seconds']:.1f} sec")
    print(f"  File size: {build_stats['video_size_mb']:.2f} MB")
    print(f"  Frame rate: {build_stats['fps']}")
    print(f"  Segments/sec: {build_stats['total_chunks'] / elapsed:.1f}")

    print("\nIndex Map Stats:")
    for key, value in build_stats['index_stats'].items():
        print(f"  {key}: {value}")

    print("\nSuccess! Clip memory ready.")
    print("\nTo interact with this memory, run:")
    print("  python examples/chat_memory.py")

if __name__ == "__main__":
    main()
