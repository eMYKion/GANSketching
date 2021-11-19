from training import networks
import os
import random
import torch
from options import get_opt, print_options
from training.dataset import create_dataloader, yield_data
from torchvision.utils import save_image

FILE_TMPL = "./data/horse_cnt/out_{:05d}.png"
IMG_LIMIT = 600

if __name__ == '__main__':
    opt, parser = get_opt()
    
    F = networks.OutputTransform(opt, process=opt.transform_fake, diffaug_policy=opt.diffaug_policy)
    F.cuda()
    print(type(F))
    
    
    dataloader_image, sampler_image = create_dataloader(opt.dataroot_image,
                                                            opt.size,
                                                            opt.batch)
    
    for i, data_image in enumerate(dataloader_image):
        if i % 60 == 0:
            print(i)
            
        if (i == IMG_LIMIT):
            break
        
        data_image = data_image.cuda()
        
        out = F(data_image, apply_aug=False)
        
        save_image(out[0].detach().cpu(), FILE_TMPL.format(i))
        
        
    
    print('done.')