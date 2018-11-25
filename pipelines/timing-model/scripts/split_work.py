import glob
import math
import pickle
import os

N_PROCS = int(os.environ['N_PROCS'])

DIRS = [path for path in glob.glob("extracted_maps/*")]

WORK_CHUNKS = []
WORK_PER_PROC = math.floor(len(DIRS) / N_PROCS)
REMAINDER = len(DIRS) % N_PROCS
print("Processing {} sets in total".format(len(DIRS)))
print("Processing {} sets per process".format(WORK_PER_PROC))
print("Also distributing remainder of {} sets".format(REMAINDER))

IDX = 0
for s in range(N_PROCS):
    next_idx = IDX + WORK_PER_PROC + (1 if REMAINDER > 0 else 0)
    REMAINDER -= 1
    WORK_CHUNKS.append(DIRS[IDX:next_idx])
    print(IDX, next_idx)
    IDX = next_idx

for i, chunk in enumerate(WORK_CHUNKS):
    pickle.dump(chunk, open('work/{}.p'.format(i), 'wb'))
