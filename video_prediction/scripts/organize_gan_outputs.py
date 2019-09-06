import glob
import os

src_dir = '/mnt/ds3lab-scratch/lming/data/min_quality/planet/quarter_cropped'
gan_dir = '/mnt/ds3lab-scratch/lming/forest-prediction/pix2pix/results/planet_pix2pix/gen_latest/images'
out_dir = '/mnt/ds3lab-scratch/lming/data/min_quality/planet/quarter_cropped_gan'
splits = ['train', 'val', 'test']

def create_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def move_split_files(split_dir, gan_dir, out_dir):
    create_dir(out_dir)
    splitfiles = glob.glob(os.path.join(split_dir, '*.png'))
    gan_template = os.path.join(gan_dir, '{filename}_fake_B.png') # outputs of the pix2pix or cyclegan
    for splitfile in splitfiles:
        filename = splitfile.split('/')[-1][:-4]
        ganfile = gan_template.format(filename=filename)
        outfile = os.path.join(out_dir, ganfile.split('/')[-1])
        if os.path.exists(ganfile):
            os.rename(ganfile, outfile)

for split in splits:
    split_dir = os.path.join(src_dir, split)
    out_split_dir = os.path.join(out_dir, split)
    move_split_files(split_dir, gan_dir, out_split_dir)
