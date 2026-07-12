import os
import multiprocessing
import time

def worker(rank, src, dst):
    if os.path.isdir(src):
        # Force a context switch/race condition
        time.sleep(1)
        try:
            os.rename(src, dst)
            print(f"Rank {rank} renamed successfully")
        except Exception as e:
            print(f"Rank {rank} failed: {e}")

if __name__ == "__main__":
    src = "test_src_dir_2"
    dst = "test_dst_dir_2"
    os.makedirs(src, exist_ok=True)
    if os.path.exists(dst):
        os.rmdir(dst)

    p1 = multiprocessing.Process(target=worker, args=(0, src, dst))
    p2 = multiprocessing.Process(target=worker, args=(1, src, dst))
    p1.start()
    p2.start()
    p1.join()
    p2.join()