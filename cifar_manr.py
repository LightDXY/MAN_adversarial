import os
import time
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from options.new_train_options import TrainOptions
from memcached_dataset import McDataset

from PIL import Image
import numpy as np
from models import cifar_generators as generators
from my_network import load_pretrained_model, load_scratch_model

def main():
    use_cuda = torch.cuda.is_available()
    opt = TrainOptions().parse()

    opt.gpu_ids = []
    opt.gpu_ids.append(0)
    print (torch.cuda.device_count())
    opt.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    
    args = vars(opt)

    netD = generators.Man_Recalibrate(3, 3, 16, norm_type='batch', act_type='relu',addition_res=True)
    netD.apply(generators.weights_init)
    
    netT = load_scratch_model('iden_resnet_14_cifar_10')
    
    if use_cuda:
        netT.cuda()
        netD.cuda()
        netT = torch.nn.DataParallel(netT, device_ids=range(torch.cuda.device_count()))
        netD = torch.nn.DataParallel(netD, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True
    netT.load_state_dict(torch.load('checkpoints/resnet14/latest_net_T.pth'))
    
    print ('basic setting:')
    print ('beta = {}'.format(opt.beta))
    print ('lr decay /iters = {}'.format(opt.lr_decay_iters))
    print ('number of class to be attacked = {}'.format(opt.set_class))
    
    eps = 2.0 * opt.max_epsilon / 255.0
    
    mean_arr = (0.5, 0.5, 0.5)
    stddev_arr = (0.5, 0.5, 0.5)
    
    base_params =[]
    for name ,para in netD.module.named_parameters():
        base_params.append(para)
        
    optimizer_G=torch.optim.Adam([{'params' : base_params}] , lr=opt.lr, betas=(opt.beta1, 0.999))
    
    optimizers=[]
    schedulers=[]
    optimizers.append(optimizer_G)
    for optimizer in optimizers:
        schedulers.append(generators.get_scheduler(optimizer, opt))
    train_dataset = McDataset(
        opt.dataroot,
        transform=transforms.Compose([
            transforms.Scale(opt.loadSize),
            transforms.RandomCrop(opt.fineSize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_arr, std=stddev_arr)
        ]) )
        
    train_loader_single = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True)
    
    train (opt,netD,netT,optimizer_G,train_loader_single, eps, optimizers, schedulers, mean_arr, stddev_arr)
    
def train (opt,netD,netT,optimizer_G,train_loader, eps, optimizers, schedulers, mean_arr, stddev_arr):
    total_steps = 0
    to_num = 0
    lossX = AverageMeter()
    lossY = AverageMeter()
    SuccR_1 = AverageMeter()
    RealR = AverageMeter()
    
    netT.eval()
    netD.train()
    L1 = torch.nn.L1Loss().cuda()
    
    L2 = torch.nn.MSELoss(reduce=False).cuda()
    cross_loss = torch.nn.CrossEntropyLoss().cuda()
    soft=torch.nn.Softmax(1)
    num_classes = 10
    onehot = torch.zeros(num_classes, num_classes)
    onehot = onehot.scatter_(1, torch.arange(0, num_classes).long().view(num_classes, 1), 1).view(num_classes, num_classes)
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        lossX.reset()
        lossY.reset()
        SuccR_1.reset()
        RealR.reset()
        timeEN=0
        timeDN=0
        timeT=0
        lr = optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)
        for i, data in enumerate(train_loader):
            to_num +=1
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            input_A = data['A']
            input_A = input_A.cuda(async=True)
            temp = input_A
            for j in range(opt.replica - 1):
                input_A = torch.cat([input_A, temp])
            image_paths = data['path']
            real_A = Variable(input_A)
            label = torch.LongTensor(input_A.size(0) ).random_(0, opt.set_class) * opt.div_coeff
            onehot_label = onehot[label]
            onehot_label = Variable(onehot_label.cuda())
            label = Variable(label.cuda())
            optimizer_G.zero_grad()
            
            adv_A = netD(real_A,onehot_label)
            
            clip_A = reconstruction(adv_A , mean_arr ,stddev_arr)
            
            logist_A = netT(real_A)
            logist_B = netT(clip_A)
            L_Y = cross_loss(logist_B ,label)
            
            L_X = L2(clip_A,real_A)
            L_X_weight = torch.mean(L_X)*opt.beta
            loss = L_Y + L_X_weight
            loss.backward()
            optimizer_G.step()
            std = Variable(torch.FloatTensor(stddev_arr).cuda().view(1, 3, 1, 1).expand_as(real_A))
            L_X_show = torch.sqrt(torch.sum(L_X * std * std *255 * 255)/len(real_A))
            
            pre_A = soft(logist_B)
            tar_A = soft(logist_A)
            _,pre_attack=torch.max(pre_A,1)
            _,pre_real=torch.max(tar_A,1)
            top1 = torch.sum(torch.eq(pre_attack.cpu().data.float(),label.cpu().data.float()).float()) / input_A.size(0)
            
            
            reduced_lossX = (L_X_show ).data.clone()
            reduced_lossY = (L_Y ).data.clone()
            
            reduced_SuccR_1 = torch.from_numpy(np.asarray( [top1 ])).float().cuda(async=True)
            
            lossX.update(reduced_lossX[0], input_A.size(0))
            lossY.update(reduced_lossY[0], input_A.size(0))
            
            SuccR_1.update(reduced_SuccR_1[0], input_A.size(0))
            

            if (i+1) % opt.print_freq == 0:
                print('[{0}][{1}/{2}]\t'
                      'LossX  {lossX.avg:.2f}\t'
                      'LossY  {lossY.avg:.2f}\t'
                      'TOP1  {SuccR_1.avg:.2f}\t'
                      'Data {batch_time:.3f} '.format(
                          epoch, i+1, len(train_loader),
                          lossX = lossX,lossY = lossY,
                          SuccR_1=SuccR_1,batch_time=(time.time() - iter_start_time) / opt.batchSize))

                          
            if total_steps % opt.save_latest_freq == 0 :
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                save_networks('latest',opt,netD,'D')
                lossX.reset()
                lossY.reset()
                SuccR_1.reset()
                RealR.reset()

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0 :
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            save_networks(epoch,opt,netD,'D')
            #model.save_networks(epoch)

        
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        for scheduler in schedulers:
            scheduler.step()
        
        lr = optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def save_networks(which_epoch,opt,net,name):
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    save_filename = '%s_net_%s.pth' % (which_epoch, name)
    save_path = os.path.join(save_dir, save_filename)
    if not os.path.exists(save_dir) :
        os.makedirs(save_dir)
    if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
        torch.save(net.cpu().state_dict(), save_path)
        net.cuda(opt.gpu_ids[0])
    else:
        torch.save(net.cpu().state_dict(), save_path)

def reconstruction(image, mean, std ):
    mean = Variable(torch.FloatTensor(mean).cuda().view(1,3,1,1).expand_as(image))
    std = Variable(torch.FloatTensor(std).cuda().view(1,3,1,1).expand_as(image))
    image = (image * 0.5 + 0.5 -mean)/std
    return image
      
if __name__ == '__main__':
    main()
