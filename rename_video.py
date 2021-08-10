import os
import sys
import re
import glob
import shutil


def handle_dextran(in_str):
    return re.sub(r'\s+without dextran\s*(?=\d)', '  ', in_str)


def unit_test():
    input1 = '100x100 micrometer  channel C  0.24 ul per min  40x  without dextran01.mp4'
    output1 = handle_dextran(input1)
    print(output1)

    input2 = '200x200 micrometer  channel B  0.72 ul per min  40x   without dextran 01.mp4'
    output2 = handle_dextran(input2)
    print(output2)


def main():
    assert len(sys.argv) == 2
    input_dir = sys.argv[1]
    if not os.path.exists(input_dir) or not os.path.isdir(input_dir):
        print('Input directory does not exist or is invalid')
        return
    
    orig_files = glob.glob(os.path.join(input_dir, '*.mp4'))
    print(f'{len(orig_files)} files to rename')
    count = 0
    for old in orig_files:
        head, old_tail = os.path.split(old)
        new_tail = handle_dextran(old_tail)
        if new_tail != old_tail:
            new = os.path.join(head, new_tail)
            shutil.move(old, new)
            count += 1
    print(f'{count} files have been renamed')


if __name__ == '__main__':
    # unit_test()
    main()