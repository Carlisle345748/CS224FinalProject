from tqdm.contrib.concurrent import process_map  # or thread_map
import time
from multiprocessing import Pool


def _foo(my_number):
	time.sleep(1)
	print(my_number)
	return my_number


if __name__ == '__main__':
	process_map(_foo, range(0, 10000), max_workers=3, chunksize=10)
	# with Pool(3) as p:
	# 	p.map(_foo, range(10000))
