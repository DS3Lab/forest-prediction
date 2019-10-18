import subprocess
import sys
import glob
import os
import argparse

'''
python run_all.py --input_dir /mnt/ds3lab-scratch/lming/data/min_quality/planet/tfrecordsinfour/gan/ --output_dir results_today/gan --model_dir logs/planet_cropped4_experiments/ours_deterministic_l1/
'''
def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir")
parser.add_argument("--output_dir")
parser.add_argument("--model_dir")
args = parser.parse_args()
directories = get_immediate_subdirectories(args.input_dir)
cmd_template = 'python scripts/generate.py --input_dir {input_dir} --dataset landsat --dataset_hparams sequence_length=5 --checkpoint {model_dir} --mode test --results_dir {output_dir} --batch_size 1'

print(os.getcwd(), 'DIRECTORY')
# cmd_template = ['python', 'scripts/generate.py', '--input_dir', '{input_dir}', '--dataset cropped', '--dataset_hparams', 'sequence_length=', '--checkpoint', 'logs/planet_cropped_gan4_experiments/ours_deterministic_l1/', '--mode test', '--results_dir', '{output_dir}', '--batch_size', '1']
env = os.environ.copy()
env['CUDA_VISIBLE_DEVICES'] = '-1'
for i in range(0, len(directories), 16):
    # p0 = subprocess.Popen(cmd)
    cmds = []
    
    for j in range(i, min(i+16, len(directories))):
        # cmd = cmd_template.copy()
        # cmd[3] = str(args.input_dir)
        # cmd[-3] = str(args.output_dir)
        # cmds.append(cmd)
        if not os.path.exists(os.path.join(args.output_dir, 'ours_deterministic_l1', directories[j])):
            print(directories[j])
    
        cmds.append(cmd_template.format(input_dir=os.path.join(args.input_dir, directories[j]), model_dir=args.model_dir, output_dir=args.output_dir))
    processes = []
    for cmd in cmds:
        print('RUN', cmd)
        p = subprocess.Popen(cmd, env=env, shell=True)
        processes.append(p)
    exit_codes = [p.wait() for p in processes]
    print('finished', i)
    
    '''
    cmd = cmd_template.format(input_dir=os.path.join(args.input_dir, directories[j+1]), output_dir=args.output_dir)
    p1 = subprocess.Popen(cmd)

    cmd = cmd_template.format(input_dir=os.path.join(args.input_dir, directories[j+2]), output_dir=args.output_dir)
    p2 = subprocess.Popen(cmd)

    cmd = cmd_template.format(input_dir=os.path.join(args.input_dir, directories[j+3]), output_dir=args.output_dir)
    p3 = subprocess.Popen(cmd)

    cmd = cmd_template.format(input_dir=os.path.join(args.input_dir, directories[j+4]), output_dir=args.output_dir)
    p4 = subprocess.Popen(cmd)

    cmd = cmd_template.format(input_dir=os.path.join(args.input_dir, directories[j+5]), output_dir=args.output_dir)
    p5 = subprocess.Popen(cmd)

    cmd = cmd_template.format(input_dir=os.path.join(args.input_dir, directories[j+6]), output_dir=args.output_dir)
    p6 = subprocess.Popen(cmd)

    cmd = cmd_template.format(input_dir=os.path.join(args.input_dir, directories[j+7]), output_dir=args.output_dir)
    p7 = subprocess.Popen(cmd)

    cmd = cmd_template.format(input_dir=os.path.join(args.input_dir, directories[j+8]), output_dir=args.output_dir)
    p8 = subprocess.Popen(cmd)

    cmd = cmd_template.format(input_dir=os.path.join(args.input_dir, directories[j+9]), output_dir=args.output_dir)
    p9 = subprocess.Popen(cmd)

    cmd = cmd_template.format(input_dir=os.path.join(args.input_dir, directories[j+10]), output_dir=args.output_dir)
    p10 = subprocess.Popen(cmd)

    cmd = cmd_template.format(input_dir=os.path.join(args.input_dir, directories[j+11]), output_dir=args.output_dir)
    p11 = subprocess.Popen(cmd)

    cmd = cmd_template.format(input_dir=os.path.join(args.input_dir, directories[j+12]), output_dir=args.output_dir)
    p12 = subprocess.Popen(cmd)

    cmd = cmd_template.format(input_dir=os.path.join(args.input_dir, directories[j+13]), output_dir=args.output_dir)
    p13 = subprocess.Popen(cmd)

    cmd = cmd_template.format(input_dir=os.path.join(args.input_dir, directories[j+14]), output_dir=args.output_dir)
    p14 = subprocess.Popen(cmd)

    cmd = cmd_template.format(input_dir=os.path.join(args.input_dir, directories[j+15]), output_dir=args.output_dir)
    p15 = subprocess.Popen(cmd)

    exit_codes = [p.wait() for p in (p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15)]
    '''
