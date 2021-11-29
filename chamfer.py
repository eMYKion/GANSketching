import argparse
import os
from tqdm import tqdm
from chamfer_distance import ChamferDistance
from training.gan_model import GANModel
from PIL import Image
# from torchvision import transforms
import cv2
import numpy as np

from training import networks
from torchvision import transforms

from options import get_opt

def main():
    parser = argparse.ArgumentParser(description='orders images by champfer distance of their sketch to reference user sketch')

    real_dir = "data/image/horse/"
    user_sketch = "data/sketch/quickdraw/single_sketch/picasso/0.png"
    out = "chamfer_nn/"
    cnt_cache = "chamfer_nn/cache"
    
    opt, _ = get_opt()
    
    chamfer_dist = ChamferDistance()
    
    IMG_DIM = (256, 256)
    THRES = 100
    
    # TODO load model weights
    # gan_model = GANModel(gansketch_opt).to('cuda')
    # img2sketch = networks.OutputTransform(opt, process=opt.transform_fake, diffaug_policy=opt.diffaug_policy)
    
    img2sketch = networks.OutputTransform(opt, process='toSketch', diffaug_policy=opt.diffaug_policy)
    img2sketch = img2sketch.setup_sketch(opt)

    size=256
    img_channel=3
    
    mean, std = [0.5 for _ in range(img_channel)], [0.5 for _ in range(img_channel)]
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std, inplace=True),
    ])
    
    print("running chamfer dist:")
    
    for fname in tqdm(os.listdir(real_dir)):
        
        if not(fname.endswith(".jpg") or fname.endswith(".png")):
            continue
            
        print("fname: ", fname)
        
        # check if cache exists
        cache_fname = "{}.npy".format(fname)
        cache_fpath = os.path.join(cnt_cache, cache_fname)
        # print("cache_fname: ", cache_fname)
        
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
            
            # adapted from https://pythonexamples.org/python-opencv-cv2-find-contours-in-image/
            
            # img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            # img = cv2.resize(img, IMG_DIM)
            
            # threshold image
            img = np.interp(sketch, (sketch.min(), sketch.max()), (0, 255)).astype(np.uint8)
            
            print(img.min(), img.max())
            ret, thresh_img = cv2.threshold(img, THRES, 255, cv2.THRESH_BINARY_INV)
            cv2.imwrite(os.path.join(out, "threshold.png"), thresh_img)
            
            # find contours
            print(thresh_img)
            contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # each contour is N_i x 1 x 2, there are C contours -> N_T x 1 x 2 -> N_T x 2
            
            #create an empty image for contours
            img_contours = np.zeros(img.shape)
            # draw the contours on the empty image
            cv2.drawContours(img_contours, contours, -1, 255, 1)
            #save image
            cv2.imwrite(os.path.join(out, "cnt.png"), img_contours)
            
            contours = np.concatenate(contours, axis=0).reshape((-1, 2))
            cache_fname = "{}.npy".format(fname)
            np.save(os.path.join(cnt_cache, cache_fname), contours)
            
        else:
            #load cache and compare
            print("cache exists, skipping")
            pass
            
        assert False, "NO1"
            
            
if __name__ == '__main__':
    main()
            
            
                
                
                
            