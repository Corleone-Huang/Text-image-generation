from __future__ import print_function

from miscc.config import cfg, cfg_from_file
from datasets import TextDataset

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np

from model2 import G_NET
from model2 import RNN_ENCODER

import torch
from nltk.tokenize import RegexpTokenizer
from miscc.utils import weights_init
from miscc.utils import mkdir_p, build_super_images2
from torch.autograd import Variable
from PIL import Image


image_per_caption = 1

def parse_args():
    parser = argparse.ArgumentParser(description='process some args')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='sample_coco2.yml', type=str)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
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

    save_dir = 'results/'
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
            fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask, cap_lens)

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
def draw_img():
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir


    args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    print("Seed: %d" % (args.manualSeed))


    special_model_path = 'Model/coco2/coco2.pth'
    dataset = TextDataset(cfg.DATA_DIR, base_size=cfg.TREE.BASE_SIZE)
    assert dataset
    
    gen_example(dataset.n_words, dataset.wordtoix, dataset.ixtoword, special_model_path)


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)


    args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    print("Seed: %d" % (args.manualSeed))


    special_model_path = 'Model/coco2/coco2.pth'
    dataset = TextDataset(cfg.DATA_DIR, base_size=cfg.TREE.BASE_SIZE)
    assert dataset
    
    gen_example(dataset.n_words, dataset.wordtoix, dataset.ixtoword, special_model_path)
    print("done!")
    

