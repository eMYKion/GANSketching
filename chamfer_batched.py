import argparse
import os
from tqdm import tqdm
from chamfer_distance import ChamferDistance
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

# takes a HxW BW sketch and returns all contours as Nx2 array
def sketch2cnt(sketch):
    # adapted from https://pythonexamples.org/python-opencv-cv2-find-contours-in-image/
    # img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
    # img = cv2.resize(img, IMG_DIM)

    # threshold image
    img = np.interp(sketch, (sketch.min(), sketch.max()), (0, 255)).astype(np.uint8)

    # print(img.min(), img.max())
    ret, thresh_img = cv2.threshold(img, THRES, 255, cv2.THRESH_BINARY_INV)
    # cv2.imwrite(os.path.join(out, "threshold.png"), thresh_img)

    # find contours
    # print(thresh_img)
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # each contour is N_i x 1 x 2, there are C contours -> N_T x 1 x 2 -> N_T x 2

    '''
    #create an empty image for contours
    img_contours = np.zeros(img.shape)
    # draw the contours on the empty image
    cv2.drawContours(img_contours, contours, -1, 255, 1)
    #save image
    cv2.imwrite(os.path.join(out, "cnt.png"), img_contours)
    '''
    
    if len(contours) == 0:
        return np.zeros((0,2))
    contours = np.concatenate(contours, axis=0).reshape((-1, 2))
    return contours

def to3dcnt(contours):
    assert contours.shape[1] == 2, "input not 2d contours"
    N, _ = contours.shape
    # N x 3 as required by chamfer distance
    if len(contours) != 0:
        contours = np.concatenate((contours, np.zeros((N, 1))), axis=1)
        assert contours.shape == (N,3), "incorrect contour shape for chamfer distance"
    return contours

def main():
    parser = argparse.ArgumentParser(description='orders images by champfer distance of their sketch to reference user sketch')

    real_dir = "data/image/horse/"
    user_sketch = "data/sketch/quickdraw/single_sketch/picasso/0.png"
    out = "chamfer_nn/"
    outf = 'chamfer.txt'
    cnt_cache = "chamfer_nn/cache"
    
    opt, _ = get_opt()
    
    #overloading batch to number of workers and name to worker index for caching
    
    chamfer_dist = ChamferDistance()
    
    # TODO load model weights
    # gan_model = GANModel(gansketch_opt).to('cuda')
    # img2sketch = networks.OutputTransform(opt, process=opt.transform_fake, diffaug_policy=opt.diffaug_policy)
    
    img2sketch = networks.OutputTransform(opt, process='toSketch', diffaug_policy=opt.diffaug_policy)
    img2sketch = img2sketch.setup_sketch(opt)
    img2sketch.eval()

    size=256
    img_channel=3
    
    mean, std = [0.5 for _ in range(img_channel)], [0.5 for _ in range(img_channel)]
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std, inplace=True),
    ])
    
    print("running chamfer dist:")
    
    
    templ_sketch = cv2.imread(user_sketch, cv2.IMREAD_GRAYSCALE)
    assert templ_sketch.shape == (256, 256)
    templ_contours = sketch2cnt(templ_sketch)
    templ_contours = to3dcnt(templ_contours)
    templ_contours = torch.from_numpy(templ_contours).unsqueeze(0).type(torch.FloatTensor)
    
    distances = []
    no_cnt = 0
    
    real_imgs = os.listdir(real_dir)
    N = len(real_imgs)
    chunk_size = int(N / opt.batch)
    w_id = int(opt.name)
    for fname in tqdm(real_imgs[chunk_size * w_id:min(chunk_size * (w_id+1), N)]):
        
        if not(fname.endswith(".jpg") or fname.endswith(".png")):
            continue
            
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