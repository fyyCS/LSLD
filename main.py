#!/usr/bin/python
#encoding:utf-8

from __future__ import print_function
import argparse
import pandas as pd
import os
import os.path as osp
import copy
import time
import random
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from collections import defaultdict
from dataloader import LLP_dataset, ToTensor, categories
from nets.net_audiovisual import MMIL_Net
from utils.eval_metrics import segment_level, event_level, print_overall_metric
import clip
import matplotlib.pylab as plt
import cv2
from PIL import Image, ImageDraw
import laion_clap
def get_LLP_dataloader(args):


    train_dataset = LLP_dataset(label=args.label_train, audio_dir=args.audio_dir,
                                video_dir=args.video_dir,
                                transform=transforms.Compose([ToTensor()]),a_smooth=args.a_smooth, v_smooth=args.v_smooth)
    val_dataset = LLP_dataset(label=args.label_val, audio_dir=args.audio_dir,
                              video_dir=args.video_dir,
                              transform=transforms.Compose([ToTensor()]))
    test_dataset = LLP_dataset(label=args.label_test, audio_dir=args.audio_dir,
                               video_dir=args.video_dir,
                               transform=transforms.Compose([ToTensor()]))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=5, pin_memory=True, sampler=None)
    # batch size = 1
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=1, pin_memory=True, sampler=None)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=1, pin_memory=True, sampler=None)


    return train_loader, val_loader, test_loader


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_random_state():
    state = {
        'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state(),
        'random_rng': random.getstate(),
        'numpy_rng': np.random.get_state()
    }
    return state

@torch.no_grad()
def getVisualCleanLabel(args, train_loader):

    refine_label = np.load(args.language, allow_pickle=True)
    label_a_refine = refine_label['label_a'].item()
    label_v_refine = refine_label['label_v'].item()
    noise_ratio = args.noise_ratio
    event_ratio = args.event_ratio

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/16", device=device)
    
    categories = ['Speech', 'Car', 'Cheering', 'Dog', 'Cat', 'Frying_(food)',
              'Basketball_bounce', 'Fire_alarm', 'Chainsaw', 'Cello', 'Banjo',
              'Singing', 'Chicken_rooster', 'Violin_fiddle', 'Vacuum_cleaner',
              'Baby_laughter', 'Accordion', 'Lawn_mower', 'Motorcycle', 'Helicopter',
              'Acoustic_guitar', 'Telephone_bell_ringing', 'Baby_cry_infant_cry', 'Blender',
              'Clapping']

    text_tokens = clip.tokenize(categories).cuda()

    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens).cuda()

    event_v = [[] for _ in range(25)]
    name_v = [[] for _ in range(25)]
    event_v_not = [[] for _ in range(25)]
    name_v_not = [[] for _ in range(25)]

    for batch_idx, (sample,name) in enumerate(train_loader):
        
        audio, video, target = sample['audio'].to('cuda'), \
                                            sample['video_s'].to('cuda'), \
                                            sample['label'].type(torch.FloatTensor).to('cuda')
        vid_s = video.permute(0, 2, 1).unsqueeze(-1)
        vid_s = F.avg_pool2d(vid_s, (8, 1)).squeeze(-1).permute(0, 2, 1)
        for i in range(audio.size(0)):
            for j in range(10):
                label_idx = np.where(label_v_refine[name[i]][j].cpu().numpy())[0]
                label_idx_not = np.where(label_v_refine[name[i]][j].cpu().numpy()==0)[0]

                for idx in label_idx:
                    event_v[idx].append(vid_s[i,j,:])
                    name_v[idx].append(name[i]+str(j))
                for idx in label_idx_not:
                    event_v_not[idx].append(vid_s[i,j,:])
                    name_v_not[idx].append(name[i]+str(j))

    for i in range(25):
        print(i)
        if len(event_v[i]) == 0:
            continue
        print(len(event_v[i]))
        event_v[i] = torch.stack(event_v[i]).reshape(-1,512).cuda()
        text = text_features[i]
        text_n = F.normalize(text, p=2, dim=-1).unsqueeze(0)
        event_n = F.normalize(event_v[i], p=2, dim=-1)
        cosine_simi = torch.mm(event_n, text_n.t()).squeeze()
        simi_all, idx_v = torch.sort(cosine_simi, dim=-1, descending=False)
        num_all = event_v[i].size(0)

        event_v_not[i] = torch.stack(event_v_not[i]).reshape(-1,512).cuda()
        event_not = F.normalize(event_v_not[i], p=2, dim=-1)
        cosine_simi_not = torch.mm(event_not, text_n.t()).squeeze()
        simi_all_n, idx_n = torch.sort(cosine_simi_not, dim=-1, descending=False)
        num_all_n = event_v_not[i].size(0)

        if simi_all_n[-1] > simi_all[0]:
            idx_noise_v = np.where(cosine_simi.cpu().numpy()<simi_all_n[-1].cpu().numpy())[0]
            for j in idx_noise_v:
                temp = name_v[i][j][:-1]
                index = name_v[i][j][-1]
                label_v_refine[temp][int(index)][i] = label_v_refine[temp][int(index)][i].float()
                # label_v_refine[temp][int(index)][i] = 0.9
                if cosine_simi[j] * event_ratio < 1 and cosine_simi[j] * event_ratio > 0:
                    label_v_refine[temp][int(index)][i] = cosine_simi[j] * event_ratio
            idx_event_v = np.where(cosine_simi_not.cpu().numpy()>simi_all[0].cpu().numpy())[0]
            for j in idx_event_v:
                temp = name_v_not[i][j][:-1]
                index = name_v_not[i][j][-1]
                label_v_refine[temp][int(index)][i] = label_v_refine[temp][int(index)][i].float()
                # label_v_refine[temp][int(index)][i] = 0.1
                if cosine_simi_not[j] * noise_ratio > 0 and cosine_simi_not[j] * noise_ratio < 1:
                    label_v_refine[temp][int(index)][i] = cosine_simi_not[j] * noise_ratio

    np.savez(args.refine_label, label_v=label_v_refine, label_a=label_a_refine)

@torch.no_grad()
def refineLabel(args, train_loader):

    label_v_refine = {}
    label_a_refine = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/16", device=device)
    
    categories = ['Speeching', 'Car', 'Cheering', 'Dog', 'Cat', 'Frying food',
            'playing basketball', 'Fire alarm ringing', 'Chainsaw', 'playing the Cello', 'playing the Banjo',
            'Singing', 'Chicken', 'playing the Violin', 'Vacuum cleaner',
            'Baby laughter', 'playing the Accordion', 'Lawn-mower', 'Motorcycle', 'Helicopter',
            'playing the guitar', 'Telephone bell ringing', 'Baby cry', 'Blender',
            'Clapping']
    
    strText = [['1', 'other'], 
            ['1 and 2', '1', '2', 'other'], 
            ['1 and 2 and 3', '1', '2', '3', 
            '1 and 2', '1 and 3', '2 and 3', 'other']]
    strText2 =[[f"This is a sound of {label}" for label in text] for text in strText] 
   
    model_clap = laion_clap.CLAP_Module(enable_fusion=False)

    model_clap.load_ckpt('./630k-audioset-best.pt') # download the default pretrained checkpoint.
    # model_clap.model.text_transform.sequential = nn.Sequential()
    for batch_idx, (sample,name) in enumerate(train_loader):
        
        audio, video, target = sample['audio'].to('cuda'), \
                                            sample['video_s'].to('cuda'), \
                                            sample['label'].type(torch.FloatTensor).to('cuda')
        Pa, Pv = sample['Pa'].type(torch.FloatTensor).to('cuda'), sample['Pv'].type(torch.FloatTensor).to('cuda')
        vid_s = video.permute(0, 2, 1).unsqueeze(-1)
        vid_s = F.avg_pool2d(vid_s, (8, 1)).squeeze(-1).permute(0, 2, 1)
        for i in range(audio.size(0)):
            label_idx_v = np.where(target[i].cpu().numpy())[0]
            label_idx_a = np.where(target[i].cpu().numpy())[0]
            label_v_refine[name[i]] = target[i].unsqueeze(0).repeat(10,1)
            label_a_refine[name[i]] = target[i].unsqueeze(0).repeat(10,1)

            if len(label_idx_v) == 1:
                text_all = [x.replace("1",categories[label_idx_v[0]]) for x in strText[len(label_idx_v)-1]]
            elif len(label_idx_v) == 2:
                text_all = [x.replace("1",categories[label_idx_v[0]]).replace("2",categories[label_idx_v[1]]) for x in strText[len(label_idx_v)-1]]
            elif len(label_idx_v) == 3:
                text_all = [x.replace("1",categories[label_idx_v[0]]).replace("2",categories[label_idx_v[1]]).replace("3",categories[label_idx_v[2]]) for x in strText[len(label_idx_v)-1]]
            elif len(label_idx_v) == 0:
                continue
            if len(label_idx_a) == 1:
                text_all_a = [x.replace("1",categories[label_idx_a[0]]) for x in strText2[len(label_idx_a)-1]]
            elif len(label_idx_a) == 2:
                text_all_a = [x.replace("1",categories[label_idx_a[0]]).replace("2",categories[label_idx_a[1]]) for x in strText2[len(label_idx_a)-1]]
            elif len(label_idx_a) == 3:
                text_all_a = [x.replace("1",categories[label_idx_a[0]]).replace("2",categories[label_idx_a[1]]).replace("3",categories[label_idx_a[2]]) for x in strText2[len(label_idx_a)-1]]
            elif len(label_idx_a) == 0:
                continue
            with torch.no_grad():
                # clap
                audio_f = F.normalize(audio[i], p=2, dim=-1)
                text_embeddings = model_clap.get_text_embedding(text_all_a, use_tensor=True).cuda()
                text_a = F.normalize(text_embeddings, p=2, dim=-1)
                cosine_simi_a = torch.matmul(text_a, audio_f.t())
                max_a, _ = cosine_simi_a.max(0)
                cosine_simi_a = cosine_simi_a.softmax(dim=0)
                _, idx_a = cosine_simi_a.max(0)

                # clip
                text_tokens = clip.tokenize(text_all).cuda()
                text_features = clip_model.encode_text(text_tokens).cuda()
                text_f = F.normalize(text_features, p=2, dim=-1)
                v_feature = F.normalize(vid_s[i], p=2, dim=-1)
                cosine_simi = torch.matmul(text_f, v_feature.t())
                cosine_simi = cosine_simi.softmax(dim=0)
                _, idx = cosine_simi.max(0)
            for t in range(10):
                if len(label_idx_v) == 1 and '1' not in strText[0][idx[t]]:
                    label_v_refine[name[i]][t,label_idx_v[0]] = 0
                if len(label_idx_a) == 1 and '1' not in strText[0][idx_a[t]]:
                    label_a_refine[name[i]][t,label_idx_a[0]] = 0
                    
                if len(label_idx_v) == 2:
                    if '1' not in strText[1][idx[t]]:
                        label_v_refine[name[i]][t,label_idx_v[0]] = 0
                    if '2' not in strText[1][idx[t]]:
                        label_v_refine[name[i]][t,label_idx_v[1]] = 0
                if len(label_idx_a) == 2: 
                    if '1' not in strText[1][idx_a[t]]:
                        label_a_refine[name[i]][t,label_idx_a[0]] = 0
                    if '2' not in strText[1][idx_a[t]] :
                        label_a_refine[name[i]][t,label_idx_a[1]] = 0
                if len(label_idx_v) == 3:
                    if '1' not in strText[2][idx[t]]:
                        label_v_refine[name[i]][t,label_idx_v[0]] = 0
                    if '2' not in strText[2][idx[t]]:
                        label_v_refine[name[i]][t,label_idx_v[1]] = 0
                    if '3' not in strText[2][idx[t]]:
                        label_v_refine[name[i]][t,label_idx_v[2]] = 0
                if len(label_idx_a) == 3:
                    if '1' not in strText[2][idx_a[t]]:
                        label_a_refine[name[i]][t,label_idx_a[0]] = 0
                    if '2' not in strText[2][idx_a[t]]:
                        label_a_refine[name[i]][t,label_idx_a[1]] = 0
                    if '3' not in strText[2][idx_a[t]]:
                        label_a_refine[name[i]][t,label_idx_a[2]] = 0
    np.savez(args.language, label_v=label_v_refine, label_a=label_a_refine)
    
def train_all_model(args, model, train_loader, optimizer, criterion, epoch):
    print("begin train_model.")
    model.train()

    refine_label = np.load(args.refine_label, allow_pickle=True)
    label_a = refine_label['label_a'].item()
    label_v = refine_label['label_v'].item()
    
    for batch_idx, (sample,name) in enumerate(train_loader):
    
        audio, video, target = sample['audio'].to('cuda'), \
                                            sample['video_s'].to('cuda'), \
                                            sample['label'].type(torch.FloatTensor).to('cuda')
        # # 128*25
        Pa, Pv = sample['Pa'].type(torch.FloatTensor).to('cuda'), sample['Pv'].type(torch.FloatTensor).to('cuda')
        
        optimizer.zero_grad()
        output, a_prob, v_prob, frame_prob = model(audio, video)
        # input min max
        output = torch.clamp(output, min=1e-7, max=1 - 1e-7)
        a_prob = torch.clamp(a_prob, min=1e-7, max=1 - 1e-7)
        v_prob = torch.clamp(v_prob, min=1e-7, max=1 - 1e-7)
    
        loss_v_all = 0
        loss_a_all = 0

        for i in range(audio.size(0)):
            loss_a_all += criterion(frame_prob[i,:,0,:], label_a[name[i]].float().cuda())
            loss_v_all += criterion(frame_prob[i,:,1,:], label_v[name[i]].float().cuda())
        loss_a_all/=audio.size(0)
        loss_v_all/=audio.size(0)

        loss1 = loss_a_all
        loss2 = loss_v_all
        loss3 = criterion(output, target)

        loss = loss1 * args.audio_weight + loss2 * args.visual_weight + \
               loss3 * args.video_weight
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.3f}\tLoss2: {:.3f}\tLoss3: {:.3f}'.
                  format(epoch, batch_idx * len(audio), len(train_loader.dataset),
                         100. * batch_idx / len(train_loader), loss1.item(),
                         loss2.item(), loss3.item()))

def eval(args, model, val_loader, set):
   
    model.eval()
    print("begin evaluate.")
    # load annotations
    df = pd.read_csv(set, header=0, sep='\t')
    df_a = pd.read_csv("./data/AVVP_eval_audio.csv", header=0, sep='\t')
    df_v = pd.read_csv("./data/AVVP_eval_visual.csv", header=0, sep='\t')
    
    id_to_idx = {id: index for index, id in enumerate(categories)}
    
    F_seg_a = []
    F_seg_v = []
    F_seg = []
    F_seg_av = []
    F_event_a = []
    F_event_v = []
    F_event = []
    F_event_av = []
   
    with torch.no_grad():
        num_vid = 0
        num_seg = 0
        num_all = 0
        
        for batch_idx, (sample,name) in enumerate(val_loader):
            num_all += 1
            audio, video, target = sample['audio'].to('cuda'), \
                                             sample['video_s'].to('cuda'), \
                                             sample['label'].to('cuda')
            output, a_prob, v_prob, frame_prob = model(audio, video)
            oa = (a_prob.cpu().detach().numpy() >= 0.5).astype(np.int_)
            ov = (v_prob.cpu().detach().numpy() >= 0.5).astype(np.int_)
            # 10*25
            Pa = frame_prob[0, :, 0, :].cpu().detach().numpy()
            Pv = frame_prob[0, :, 1, :].cpu().detach().numpy()
           
            Pa = (Pa >= 0.5).astype(np.int_) * np.repeat(oa, repeats=10, axis=0)
            Pv = (Pv >= 0.5).astype(np.int_) * np.repeat(ov, repeats=10, axis=0)
           
            GT_a = np.zeros((25, 10))
            GT_v = np.zeros((25, 10))

            df_vid_a = df_a.loc[df_a['filename'] == df.loc[batch_idx, :][0]]
            filenames = df_vid_a["filename"]
            events = df_vid_a["event_labels"]
            onsets = df_vid_a["onset"]
            offsets = df_vid_a["offset"]
            num = len(filenames)
            if num > 0:
                for i in range(num):
                    x1 = int(onsets[df_vid_a.index[i]])
                    x2 = int(offsets[df_vid_a.index[i]])
                    event = events[df_vid_a.index[i]]
                    idx = id_to_idx[event]
                    GT_a[idx, x1:x2] = 1

            df_vid_v = df_v.loc[df_v['filename'] == df.loc[batch_idx, :][0]]
            filenames = df_vid_v["filename"]
            events = df_vid_v["event_labels"]
            onsets = df_vid_v["onset"]
            offsets = df_vid_v["offset"]
            num = len(filenames)
            if num > 0:
                for i in range(num):
                    x1 = int(onsets[df_vid_v.index[i]])
                    x2 = int(offsets[df_vid_v.index[i]])
                    event = events[df_vid_v.index[i]]
                    idx = id_to_idx[event]
                    GT_v[idx, x1:x2] = 1
            # GT_av
            GT_av = GT_a * GT_v
            GT_a_t = np.transpose(GT_a)
            GT_v_t = np.transpose(GT_v)
            for i in range(10):
                label_idx_a = np.where(GT_a_t[i])[0]
                label_idx_v = np.where(GT_v_t[i])[0]
                if not np.array_equal(label_idx_a, label_idx_v):
                    num_seg+=1
                    
            for i in range(10):
                label_idx_a = np.where(GT_a_t[i])[0]
                label_idx_v = np.where(GT_v_t[i])[0]
                if not np.array_equal(label_idx_a, label_idx_v):
                    num_vid+=1
                    break
            SO_a = np.transpose(Pa)
            SO_v = np.transpose(Pv)
            SO_av = SO_a * SO_v

            f_a, f_v, f, f_av = segment_level(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av)
            F_seg_a.append(f_a)
            F_seg_v.append(f_v)
            F_seg.append(f)
            F_seg_av.append(f_av)

            # event-level F1 scores
            f_a, f_v, f, f_av = event_level(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av)
            F_event_a.append(f_a)
            F_event_v.append(f_v)
            F_event.append(f)
            F_event_av.append(f_av)
    
    print(num_all)
    print('unatch segment percent:{}\n'.format(num_seg/(10*num_all)))
    print('unatch video percent:{}\n'.format(num_vid/num_all))
    audio_segment_level, visual_segment_level, av_segment_level, avg_type, avg_event, \
        audio_event_level, visual_event_level, av_event_level, avg_type_event, avg_event_level \
        = print_overall_metric(F_seg_a, F_seg_v, F_seg, F_seg_av, F_event_a, F_event_v, F_event, F_event_av)
    return audio_segment_level, visual_segment_level, av_segment_level, avg_type, avg_event, \
        audio_event_level, visual_event_level, av_event_level, avg_type_event, avg_event_level
        
        
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Implementation of Audio-Visual Video Parsing')
    parser.add_argument("--audio_dir", type=str, default='./data/feats/clap_audioset/', help="audio dir")
    parser.add_argument("--video_dir", type=str, default='./data/feats/clip_16/', help="video dir")
    parser.add_argument("--label_train", type=str, default="./data/AVVP_train.csv", help="weak train csv file")
    parser.add_argument("--label_val", type=str, default="./data/AVVP_val_pd.csv", help="weak val csv file")
    parser.add_argument("--label_test", type=str, default="./data/AVVP_test_pd.csv", help="weak test csv file")
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train')
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam'], default='adam')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--lr_step_size', type=int, default=6)
    parser.add_argument('--lr_gamma', type=float, default=0.25)
    parser.add_argument('--seed', type=int, default=6, help='random seed')
    parser.add_argument("--mode", type=str, default='train_noise_estimator',
                        help="with mode to use")
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--audio_weight', type=float, default=3.0)
    parser.add_argument('--visual_weight', type=float, default=1.0)
    parser.add_argument('--video_weight', type=float, default=1.0)
    parser.add_argument('--nce_weight', type=float, default=1.0)
    parser.add_argument('--clamp', type=float, default=1e-7)
    parser.add_argument('--nce_smooth', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=0.2, help='feature temperature number')
    parser.add_argument('--log_interval', type=int, default=10, help='how many batches for logging training status')
    parser.add_argument('--log_file', type=str, help="./log/estimator")
    parser.add_argument('--save_model', type=str, default='false', help='whether to save model')
    parser.add_argument('--a_smooth', type=float, default=1.0)
    parser.add_argument('--v_smooth', type=float, default=0.9)
    parser.add_argument("--model_save_dir", type=str, default='./ckpt/', help="model save dir")
    parser.add_argument("--refine_label", type=str, default='./refine_label/final_label.npz', help="final label")
    parser.add_argument('--event_ratio', type=float, default=4)
    parser.add_argument('--noise_ratio', type=float, default=0.4)
    parser.add_argument("--language", type=str, default='./refine_label/denoised_label.npz', help="denoised label")
    parser.add_argument("--checkpoint", type=str, default='MMIL_Net', help="save model name")
    args = parser.parse_args()

    # print parameters
    print('----------------args-----------------')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print('----------------args-----------------')
    cur = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print('current time: {}'.format(cur))

    set_random_seed(args.seed)
    save_model = args.save_model == 'true'
    os.makedirs(args.model_save_dir, exist_ok=True)


    model = MMIL_Net(args.num_layers).to('cuda')

    start = time.time()

    if args.mode == 'label_denoise':
        train_loader, val_loader, test_loader = get_LLP_dataloader(args)    
        if not os.path.exists(args.language):
            refineLabel(args, train_loader)
        if not os.path.exists(args.refine_label):
            getVisualCleanLabel(args, train_loader)

    elif args.mode == 'train_model':
        args.with_ca = True
        train_loader, val_loader, test_loader = get_LLP_dataloader(args)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

        criterion = nn.BCELoss()

        best_F = 0
        best_model = None

        for epoch in range(1, args.epochs+1):
            train_all_model(args, model, train_loader, optimizer, criterion, epoch=epoch)
            scheduler.step()
            print("Validation Performance of Epoch {}:".format(epoch))

            audio_seg, visual_seg, av_seg, avg_type_seg, avg_event_seg, \
            audio_eve, visual_eve, av_eve, avg_type_eve, avg_event_eve \
                = eval(args, model, val_loader, args.label_val)
            
            if audio_eve >= best_F:
                best_F = audio_eve
                best_model = copy.deepcopy(model)
                best_epoch = epoch
                if save_model:
                    state_dict = get_random_state()
                    state_dict['model'] = model.state_dict()
                    state_dict['optimizer'] = optimizer.state_dict()
                    state_dict['scheduler'] = scheduler.state_dict()
                    state_dict['epochs'] = args.epochs
                    torch.save(state_dict, osp.join(args.model_save_dir, args.checkpoint2))
        optimizer.zero_grad()
        model = best_model
        print("Test the best model:")
        print(best_epoch)
        eval(args, model, test_loader, args.label_test)

    elif args.mode == 'test_LSLD':
        dataset = args.label_test
        test_dataset = LLP_dataset(label=dataset,
                                   audio_dir=args.audio_dir, video_dir=args.video_dir, 
                                   transform=transforms.Compose([ToTensor()]))
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        resume = torch.load(osp.join(args.model_save_dir, args.checkpoint))
        model.load_state_dict(resume['model'])

        eval(args, model, test_loader, dataset)

    end = time.time()
    print(f'duration time {(end - start) / 60} mins.'.format(end, start))
    cur = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print(f'current time: {cur}'.format(cur))


if __name__ == '__main__':
    main()
