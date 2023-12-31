import shutil
import subprocess
import glob
from tqdm import tqdm
import numpy as np
import os
import os.path as osp
import argparse
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
import pretrainedmodels
from pretrainedmodels import utils
from mmcv import ProgressBar
import random
import clip

C, H, W = 3, 224, 224
device = "cuda" if torch.cuda.is_available() else "cpu"

def extract_feats(params, model, preprocess):
    global C, H, W
    model.eval()
    dir_fc = os.path.join(os.getcwd(), params['output_dir'])
    if not os.path.isdir(dir_fc):
        os.mkdir(dir_fc)

    video_list = os.listdir(params['video_path'])
    random.shuffle(video_list)
    nn = 0
    process_videos = list()
    pb = ProgressBar(len(video_list))
    pb.start()
    for video in video_list:
        outfile = os.path.join(dir_fc, video + '.npy')
        nn = nn + 1
        dst = video

        image_list = sorted(glob.glob(os.path.join(params['video_path'], dst, '*.jpg')))
        print(len(image_list))
        number = len(image_list)
        samples = np.round(np.linspace(
            0, len(image_list) - 1, params['n_frame_steps']))
        # for sample in samples:
        #     print(image_list[min(int(sample), number - 1)])
        try:
            image_list = [image_list[min(int(sample), number - 1)] for sample in samples]
        except:
            pb.update()
            continue
        images = torch.zeros((len(image_list), C, H, W))
        # print(images.size())
        i = 0
        for iImg in range(len(image_list)):
            # img = load_image_fn(image_list[iImg])
            img = preprocess(Image.open(image_list[iImg])).unsqueeze(0).to(device)
            images[iImg] = img

        with torch.no_grad():
            
            fc_feats = model.encode_image(images.cuda())
            
        # print("\nfc_feats.size():")    
        print(fc_feats.size())
        img_feats = fc_feats.cpu().numpy()
        print(img_feats.shape)
        # Save the inception features
        if not os.path.exists(outfile):
            np.save(outfile, img_feats)
        # cleanup
        #shutil.rmtree(dst)
        # print(nn)
        process_videos.append(video)
        pb.update()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", dest='gpu', type=str, default='1',
                        help='Set CUDA_VISIBLE_DEVICES environment variable, optional')
    parser.add_argument("--output_dir", dest='output_dir', type=str,
                        default='../data/feats/clip_16', help='directory to store features')
    parser.add_argument("--n_frame_steps", dest='n_frame_steps', type=int, default=80,
                        help='how many frames to sampler per video')

    parser.add_argument("--video_path", dest='video_path', type=str,
                        default='../data/LLP_dataset/frame', help='path to video dataset')
    parser.add_argument("--model", dest="model", type=str, default='resnet152',
                        help='the CNN model you want to use to extract_feats')

    # parser.add_argument('--log_dir', dest='log_dir', type=str, default='log/extract_rgb_feat')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    params = vars(args)
    if params['model'] == 'inception_v3':
        C, H, W = 3, 299, 299
        model = pretrainedmodels.inceptionv3(pretrained='imagenet')
        load_image_fn = utils.LoadTransformImage(model)

    elif params['model'] == 'resnet152':
        C, H, W = 3, 224, 224
        # model = pretrainedmodels.resnet152(pretrained='imagenet')
        
        model, preprocess = clip.load('ViT-B/16', device)
        # load_image_fn = utils.LoadTransformImage(model)

    elif params['model'] == 'vgg19_bn':
        C, H, W = 3, 224, 224
        model = pretrainedmodels.vgg19_bn(pretrained='imagenet')
        load_image_fn = utils.LoadTransformImage(model)
    elif params['model'] == 'inception_v4':
        C, H, W = 3, 299, 299
        model = pretrainedmodels.inceptionv4(
            num_classes=1000, pretrained='imagenet')
        load_image_fn = utils.LoadTransformImage(model)
    elif params['model'] == 'nasnetalarge':
        C, H, W = 3, 299, 299
        model = pretrainedmodels.inceptionv4(
            num_classes=1000, pretrained='imagenet')
        load_image_fn = utils.LoadTransformImage(model)

    else:
        print("doesn't support %s" % (params['model']))
    model = model.cuda()
    extract_feats(params, model, preprocess)