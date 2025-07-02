#!/usr/bin/env python3
"""
Football data memory demo using chat_with_memory
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framerecall import FrameRecallEncoder, chat_with_memory

# Knowledge input: international and club football history
football_facts = [
    "Brazil tops the World Cup winners list with 5 championships (1958, 1962, 1970, 1994, 2002).",
    "Argentina clinched the 2022 title in Qatar, fulfilling Lionel Messi’s lifelong ambition.",
    "Pelé remains the only athlete to secure 3 World Cups and has over a thousand goals to his name.",
    "Maradona is remembered for the 'Hand of God' and his solo effort against England in 1986.",
    "Messi, a 7-time Ballon d'Or recipient, lifted the World Cup trophy in 2022.",
    "Cristiano Ronaldo, awarded Ballon d'Or 5 times, leads Champions League scoring charts.",
    "Real Madrid holds the record with 14 Champions League triumphs.",
    "Barcelona's era under Guardiola reshaped tactics with its iconic tiki-taka approach.",
    "Manchester United boasts 13 Premier League trophies, the highest total in the league.",
    "Arsenal’s 2003-04 squad completed the season undefeated – 'The Invincibles'.",
    "Leicester City defied 5000-1 odds to claim the Premier League title in 2015-16.",
    "Liverpool overturned a 3-0 deficit in the 2005 Champions League final vs AC Milan – 'Istanbul Miracle'.",
    "Uruguay hosted and won the inaugural World Cup in 1930.",
    "Only 8 countries have won the tournament: Brazil, Germany, Italy, Argentina, France, Uruguay, Spain, and England.",
    "Miroslav Klose holds the World Cup scoring record with 16 goals for Germany.",
    "Camp Nou in Barcelona is Europe’s largest football stadium with space for 99,354 fans.",
    "Old Trafford, Manchester United’s home, is famously known as the 'Theatre of Dreams'.",
    "The rivalry between Real Madrid and Barcelona, 'El Clásico', is football’s biggest clash.",
    "For the first time, the 2026 edition will host 48 competing nations.",
    "Neymar’s move to PSG from Barcelona in 2017 for €222 million remains the priciest transfer ever."
]

# Output paths
video_output = "output/soccer_memory.mp4"
index_output = "output/soccer_memory_index.json"

# Prepare folder for chat sessions
os.makedirs("output/soccer_chat", exist_ok=True)

# Build video memory
encoder = FrameRecallEncoder()
encoder.add_chunks(football_facts)
encoder.build_video(video_output, index_output)
print(f"Soccer memory video created at: {video_output}")

# Retrieve API credentials
api_key = "your-api-key-here"
if not api_key:
    print("\nℹ️  Please provide your OPENAI_API_KEY for complete answers.")
    print("Otherwise, only memory snippets will be displayed.\n")

# Launch interactive Q&A session
chat_with_memory(video_output, index_output, api_key=api_key, session_dir="output/soccer_chat")
