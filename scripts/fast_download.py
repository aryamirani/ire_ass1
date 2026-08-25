import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

URL = "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_testset.zip"
TARGET = Path("data/raw/ebnerd/ebnerd_testset.zip")
NUM_CHUNKS = 16

def get_file_size(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req) as resp:
        return int(resp.headers.get("Content-Length", 0))

def download_chunk(url, start, end, chunk_file):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req) as resp, open(chunk_file, "wb") as f:
        while True:
            buf = resp.read(1024 * 1024)
            if not buf:
                break
            f.write(buf)
    return chunk_file

def main():
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    total_size = get_file_size(URL)
    print(f"Total size: {total_size / (1024*1024):.2f} MB")
    
    chunk_size = total_size // NUM_CHUNKS
    ranges = []
    chunk_files = []
    
    for i in range(NUM_CHUNKS):
        start = i * chunk_size
        end = total_size - 1 if i == NUM_CHUNKS - 1 else (i + 1) * chunk_size - 1
        c_file = TARGET.parent / f"ebnerd_testset.part{i:02d}"
        ranges.append((start, end, c_file))
        chunk_files.append(c_file)
        
    print(f"Starting {NUM_CHUNKS} parallel download threads...")
    start_t = time.time()
    with ThreadPoolExecutor(max_workers=NUM_CHUNKS) as executor:
        futures = [
            executor.submit(download_chunk, URL, start, end, c_file)
            for start, end, c_file in ranges
        ]
        for f in futures:
            f.result()
            print(f"Completed chunk in {time.time() - start_t:.1f}s")
            
    print("All chunks downloaded! Merging into final zip...")
    with open(TARGET, "wb") as out_f:
        for c_file in chunk_files:
            with open(c_file, "rb") as in_f:
                while True:
                    buf = in_f.read(8 * 1024 * 1024)
                    if not buf:
                        break
                    out_f.write(buf)
            c_file.unlink()
            
    print(f"Successfully downloaded and assembled {TARGET} ({os.path.getsize(TARGET) / (1024*1024):.2f} MB) in {time.time() - start_t:.1f}s!")

if __name__ == "__main__":
    main()
