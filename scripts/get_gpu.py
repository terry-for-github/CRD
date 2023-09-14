import os
import sys
import time
import argparse
import random


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def get_gpus_info():
    if not os.path.exists("~/.tmp_free_gpus") or time.time() - os.path.getmtime("~/.tmp_free_gpus") > 2:
        os.system('nvidia-smi -q -d Memory | grep -A4 GPU | grep Used > ~/.tmp_free_gpus')
    with open(os.path.expanduser('~/.tmp_free_gpus') , 'r') as f:
        lines = f.readlines()
        return [line.split()[2] == '0' for line in lines]
    

def fetch_gpus():
    gpus_info = get_gpus_info()
    result = []
    for idx in range(len(gpus_info)):
        if gpus_info[idx]:
            result.append(idx)
    return result


def check_gpus(gpu_idx):
    if isinstance(gpu_idx, str) and gpu_idx.isdigit():
        gpu_idx = int(gpu_idx)
    if isinstance(gpu_idx, int):
        gpu_idx = [gpu_idx]
    gpus_info = get_gpus_info()
    if max(gpu_idx) >= len(gpus_info):
        raise IndexError('Gpu index out of range!')
    for i in gpu_idx:
        if not gpus_info[i]:
            return False
    return True


if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = find_gpus(nums_needed=1, mem_min=4000, time_sleep=60)
    parser = argparse.ArgumentParser('')
    parser.add_argument('-n', '--name', type=str, default='')
    parser.add_argument('-m', '--mode', type=str, default='random')
    parser.add_argument('-g', '--gpu_num', type=int, default=1)
    parser.add_argument('-l', '--gpu_list', type=int, nargs='+', default=0)
    parser.add_argument('-t', '--inter_sec', type=int, default=30)
    args = parser.parse_args()
    while True:
        if args.mode == 'random':
            avail_gpus = fetch_gpus()
            if len(avail_gpus) >= args.gpu_num:
                avail_gpus = ','.join(map(str, avail_gpus[:args.gpu_num]))
                eprint(args.name, os.getpid(), time.time(), ' Get available GPUs: {}'.format(avail_gpus))
                print(avail_gpus)
                break
            else:
                eprint(args.name, os.getpid(), time.time(), ' Only {} available gpus, need {}'.format(len(avail_gpus), args.gpu_num))
        elif args.mode == 'specify':
            if check_gpus(args.gpu_list):
                eprint(args.name, os.getpid(), time.time(), ' Get available GPUs: {}'.format(args.gpu_list))
                print(','.join(map(str, args.gpu_list)))
                break
            else:
                eprint(args.name, os.getpid(), time.time(), ' Unsatisify {} gpus'.format(args.gpu_list))
        time.sleep(args.inter_sec + random.randint(0, 10))
                
