import argparse
import os
from tqdm import tqdm
from training.gan_model import GANModel
from PIL import Image
# from torchvision import transforms
import cv2
import numpy as np
import torch

from training import networks
from torchvision import transforms

from options import get_opt

IMG_DIM = (256, 256)
THRES = 100

def main():
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str)
    parser.add_argument("--dataroot_sketch", type=str, required=True, help="root to the sketch dataset")
    parser.add_argument("--dataroot_image", type=str, default=None, help="root to the image dataset for image regularization")

    real_dir = "data/image/horse/"
    user_sketch = "data/sketch/quickdraw/single_sketch/picasso/0.png"
    out = "chamfer_nn/"
    outf = 'chamfer.txt'
    cnt_cache = "chamfer_nn/cache"
    
    if ()
    
    args = parser.parse_args()
    with nnds_fname as open(nnds_file, "r"):
        lines = nnds_fname.readlines()
        for line in tqdm(lines[0:int(len(lines)/5)]):
            img_fname = line.split(',')[0]
            
            
            
        # print("fname: ", fname)
        
        # check if cache exists
        cache_fname = "{}.npy".format(fname)
        cache_fpath = os.path.join(cnt_cache, cache_fname)
        # print("cache_fname: ", cache_fname)
        
        contours = None
        
        # if cache does not exist, make it
        if (not os.path.exists(cache_fpath)):
            # print("cache does not exist, creating contours")
        
            fpath = os.path.join(real_dir, fname)
            pil_img = Image.open(fpath).convert('RGB')
            
            tf_img = transform(pil_img)
            
            tf_img = tf_img.unsqueeze(0)
            # print("tf_img shape: {}".format(tf_img.shape))
            
            sketch = img2sketch(tf_img)
            # print("sketch shape: {}".format(sketch.shape))
            
            sketch = sketch.squeeze().detach().cpu().numpy()
            
            contours = sketch2cnt(sketch)
            
            np.save(cache_fpath, contours)
        else:
            contours = np.load(cache_fpath)
            # print("contours shape: ", contours.shape)
            
        # now cache definitely created
        contours = to3dcnt(contours)
        if contours.shape[0] == 0:
            no_cnt += 1
            continue
        
        contours = torch.from_numpy(contours).unsqueeze(0).type(torch.FloatTensor)
        
        d1, d2 = chamfer_dist(contours, templ_contours)
        loss = (torch.mean(d1)) + (torch.mean(d2))
        
        distances.append((fname, loss))
        # assert False, "loss: {}".format(loss)
        
    print('sorting distances...')
    distances = sorted(distances, key=lambda x: x[1]) # sort files according to chamfer distance
    print('writing sorted dataset to {} ...'.format(outf))
    with open(os.path.join(out, outf), 'w') as f:
        f.writelines(map(lambda x: "{}, {:.4f}\n".format(x[0], x[1]), distances))
    
    print("there were {} real images with no sketch contours".format(no_cnt))

if __name__ == '__main__':
    with torch.no_grad():
        main()
