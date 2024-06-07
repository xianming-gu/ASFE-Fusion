import os
import sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import warnings
from tqdm import trange, tqdm
import torch.utils.data as data
import torchvision.transforms as transforms
# from loss import grad_loss, ints_loss
from lossfn_asfe import compute_loss
from dataloader_HMIF import TrainData
from args_setting import args

# from net_DFTv2P_fusion_strategy import net_pyramid as net
from network import ASFEFusion

# from FourierBranch_ab import net_pyramid as net


warnings.filterwarnings("ignore")


def setup_seed(seed=3407):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def Mytrain(model_pretrain=None):
    # 设置随机数种子
    setup_seed()
    model_path = './modelsave/' + args.task + '/'
    os.makedirs(model_path, exist_ok=True)
    # os.makedirs('./modelsave')

    lr = args.lr

    # device handling
    if args.DEVICE == 'cpu':
        device = 'cpu'
    else:
        device = args.DEVICE
        # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # prepare model folder
    temp_dir = './temp/' + args.task
    os.makedirs(temp_dir, exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor()])

    train_set = TrainData(transform=transform)
    train_loader = data.DataLoader(train_set,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   drop_last=True,
                                   num_workers=1,
                                   pin_memory=True)
    # model = net(img_size=args.imgsize, dim=256)
    # print('train datasets lenth:', len(train_loader))
    model = ASFEFusion()
    if model_pretrain is not None:
        model.load_state_dict(torch.load(model_pretrain, map_location=args.DEVICE))

    # prepare the model for training and send to device
    model.to(device)
    model.train()

    loss_plt = []
    for epoch in range(0, args.epoch):
        loss_mean = []
        for idx, datas in enumerate(tqdm(train_loader, desc='[Epoch--%d]' % (epoch + 1))):
            # for idx, datas in tqdm(train_loader):

            # print(len(data))
            img1, img2 = datas
            # 训练模型
            model, img_fusion, loss_per_img = train(model, img1, img2, lr, device)
            loss_mean.append(loss_per_img)

        # print loss
        sum_list = 0
        for item in loss_mean:
            sum_list += item
        sum_per_epoch = sum_list / len(loss_mean)
        print('\tLoss:%.5f' % sum_per_epoch)
        loss_plt.append(sum_per_epoch.detach().cpu().numpy())

        # save info to txt file
        strain_path = temp_dir + '/temp_loss.txt'
        Loss_file = 'Epoch--' + str(epoch + 1) + '\t' + 'Loss:' + str(sum_per_epoch.detach().cpu().numpy())
        with open(strain_path, 'a') as f:
            f.write(Loss_file + '\r\n')

        # save model
        if (epoch + 1) % 2000 == 0 or epoch + 1 == args.epoch:
            torch.save(model.state_dict(), model_path + str(epoch + 1) + '_' + 'ASFEFusion.pth')
            print('model save in %s' % './modelsave/' + args.task)

    # 输出损失函数曲线
    plt.figure()
    x = range(0, args.epoch)  # x和y的维度要一样
    y = loss_plt
    plt.plot(x, y, 'r-')  # 设置输出样式
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(model_path + '/loss.png')  # 保存训练损失曲线图片
    plt.show()  # 显示曲线


def train(model, img1, img2, lr, device):
    model.to(device)
    model.train()

    img1 = img1.to(device)
    img2 = img2.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr)

    img1 = (img1 - img1.min()) / (img1.max() - img1.min())
    img2 = (img2 - img2.min()) / (img2.max() - img2.min())

    img_fusion = model(img1, img2)
    # img_fusion = img_fusion.cpu()
    img_fusion = img_fusion.to(device)

    img_cat = torch.cat([img1, img2], dim=1)
    # loss_total = compute_loss(img_fusion, img_cat, img_s, img_f)
    loss_total = compute_loss(img_fusion, img_cat, img1, img2)
    # loss_total = torch.from_numpy(loss_total.numpy())

    opt.zero_grad()
    loss_total.backward()
    opt.step()

    return model, img_fusion, loss_total


if __name__ == '__main__':
    Mytrain(model_pretrain=None)
