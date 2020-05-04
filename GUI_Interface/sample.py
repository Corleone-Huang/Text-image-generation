# 本函数用于从训练好的生成器中采样
# Warning: 注意修改sample.yml文件，加载G模型必须使用和训练G相同的相关模型参数！！！
# author: Huang Mengqi
# 格式：python sample.py --gpu 1

# -*- encoding: utf-8 -*-
'''
@File    :   sample.py
@Author  :   Huang Mengqi 
@Version :   1.0
@Contact :   huangmq@mail.ustc.edu.cn
@Last Modified :   2020/04/26 09:56:23
'''

# here put the import lib

import numpy as np 
import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import glob
from PIL import Image

import torch
from torch.autograd import Variable
import torchvision.transforms as transforms

from model import G_NET
from miscc.utils import weights_init
from miscc.utils import mkdir_p, build_super_images2
from model import RNN_ENCODER
from miscc.config import cfg
from datasets import TextDataset
from miscc.config import cfg, cfg_from_file
from nltk.tokenize import RegexpTokenizer

from sample2 import gen_example as gen_example2

image_per_caption = 1
# 每个caption生成多少张图片

def parse_args():
    parser = argparse.ArgumentParser(description='Sample from pretrained generator')
    parser.add_argument('--netG_path', help='the path to the pretrained generator', type=str, default='')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='sample.yml', type=str)
    parser.add_argument('--manualSeed', type=int, help='manual seed', default=None)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    args = parser.parse_args()
    return args


def gen_example(n_words, wordtoix, ixtoword, model_dir):
    '''generate images from example sentences'''
    # filepath = 'example_captions.txt'
    filepath = 'caption.txt'
    data_dic = {}
    with open(filepath, "r") as f:
        filenames = f.read().split('\n')

        captions = []
        cap_lens = []

        for sent in filenames:
            if len(sent) == 0:
                continue
            sent = sent.replace("\ufffd\ufffd", " ")
            tokenizer = RegexpTokenizer(r'\w+')
            tokens = tokenizer.tokenize(sent.lower())
            if len(tokens) == 0:
                print('sentence token == 0 !')
                continue

            rev = []
            for t in tokens:
                t = t.encode('ascii', 'ignore').decode('ascii')
                if len(t) > 0 and t in wordtoix:
                    rev.append(wordtoix[t])
            captions.append(rev)
            cap_lens.append(len(rev))

        max_len = np.max(cap_lens)
        sorted_indices = np.argsort(cap_lens)[::-1]
        cap_lens = np.asarray(cap_lens)
        cap_lens = cap_lens[sorted_indices]
        cap_array = np.zeros((len(captions), max_len), dtype='int64')

        for i in range(len(captions)):
            idx = sorted_indices[i]
            cap = captions[idx]
            c_len = len(cap)
            cap_array[i, :c_len] = cap
        # key = name[(name.rfind('/') + 1):]
        key = 0
        data_dic[key] = [cap_array, cap_lens, sorted_indices]
    
    # algo.gen_example(data_dic)
    text_encoder = RNN_ENCODER(n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    state_dict = torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
    text_encoder.load_state_dict(state_dict)
    print('Load text encoder from:', cfg.TRAIN.NET_E)
    text_encoder.eval()

    netG = G_NET()
    netG.apply(weights_init)
    # netG.cuda()
    netG.eval()
    state_dict = torch.load(model_dir, map_location=lambda storage, loc: storage)
    netG.load_state_dict(state_dict)
    print('Load G from: ', model_dir)

    save_dir = 'results'
    mkdir_p(save_dir)
    for key in data_dic:
        captions, cap_lens, sorted_indices = data_dic[key]

        batch_size = captions.shape[0]
        nz = cfg.GAN.Z_DIM

        with torch.no_grad():
            captions = Variable(torch.from_numpy(captions))
            cap_lens = Variable(torch.from_numpy(cap_lens))

            # captions = captions.cuda()
            # cap_lens = cap_lens.cuda()
        
        for i in range(image_per_caption):  # 16
            with torch.no_grad():
                noise = Variable(torch.FloatTensor(batch_size, nz))
                # noise = noise.cuda()
            
            # (1) Extract text embeddings
            hidden = text_encoder.init_hidden(batch_size)
            words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
            mask = (captions == 0)
            # (2) Generate fake images
            noise.data.normal_(0, 1)
            fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)

            cap_lens_np = cap_lens.data.numpy()

            for j in range(batch_size):
                save_name = '%s/%d_%d' % (save_dir, i, sorted_indices[j])
                for k in range(len(fake_imgs)):
                    im = fake_imgs[k][j].data.cpu().numpy()
                    im = (im + 1.0) * 127.5
                    im = im.astype(np.uint8)
                    # print('im', im.shape)
                    im = np.transpose(im, (1, 2, 0))
                    # print('im', im.shape)
                    im = Image.fromarray(im)
                    fullpath = '%s_g%d.png' % (save_name, k)
                    im.save(fullpath)
                
                for k in range(len(attention_maps)):
                    if len(fake_imgs) > 1:
                        im = fake_imgs[k + 1]
                    else:
                        im = fake_imgs[0]
                    attn_maps = attention_maps[k]
                    att_sze = attn_maps.size(2)
                    img_set, sentences = \
                        build_super_images2(im[j].unsqueeze(0),
                                            captions[j].unsqueeze(0),
                                            [cap_lens_np[j]], ixtoword,
                                            [attn_maps[j]], att_sze)
                    if img_set is not None:
                        im = Image.fromarray(img_set)
                        fullpath = '%s_a%d_attention.png' % (save_name, k)
                        im.save(fullpath)

                



def draw_picture(label):
    if label == 'bird':
        cfg_from_file('sample.yml')
        special_model_path = 'Model/bird_2.pth'
        special_model_path = 'Model/netG_epoch_575.pth'
    elif  label == 'coco':
        cfg_from_file('sample_coco.yml')
        special_model_path = 'Model/coco/coco_2.pth'
        #special_model_path = 'Model/coco/netG_epoch_55.pth'


    manualSeed = random.randint(1, 10000)
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    print('generating images for customized captions ... ')
    dataset = TextDataset(cfg.DATA_DIR, base_size=cfg.TREE.BASE_SIZE)
    assert dataset

    # print(dataset.n_words)
    if label == 'bird' or 'coco':
        gen_example(dataset.n_words, dataset.wordtoix, dataset.ixtoword,special_model_path)






if __name__ == "__main__":
    args = parse_args()

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)


    
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    
    # print('Using config:')
    # pprint.pprint(cfg)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    print('generating images for customized captions ... ')
    #special_model_path = 'Model/coco/netG_epoch_55.pth'
    special_model_path = 'Model/bird_2.pth'


    
    dataset = TextDataset(cfg.DATA_DIR, base_size=cfg.TREE.BASE_SIZE)
    assert dataset

    # print(dataset.n_words)
    gen_example(dataset.n_words, dataset.wordtoix, dataset.ixtoword, special_model_path)

    print("Done!")


