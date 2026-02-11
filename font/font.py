import numpy as np
import os
from pathlib import Path

character_index = {}

def load_character_index(font_dir=None):
    if font_dir is None:
        # Default to font directory relative to this script
        font_dir = Path(__file__).parent
    else:
        font_dir = Path(font_dir)
    
    files = [f for f in os.listdir(font_dir) if f.endswith(".csv")]
    for file in files:
        data = open(font_dir / file, "r").read()
        data = [[[255., 255., 255.] if c == '1' else [0., 0., 0.] for c in line.split(',')] for line in data.split("\n")]
        character_index[chr(int(file[5:-4]))] = (np.array(data), len(data[0]), len(data))

def put_text(frame, text: str, start):
    column_start = start[1]
    cursor = start
    for c in text:
        if c == "\n":
            cursor = [cursor[0]+6, column_start]
            continue
        letter = character_index[c]
        frame[cursor[0]:cursor[0]+letter[2], cursor[1]:cursor[1]+letter[1]] = letter[0]
        cursor[1] += letter[1]+1
