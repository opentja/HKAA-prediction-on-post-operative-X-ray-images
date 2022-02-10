import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train import TrainTestPrediction
import argparse
import torch
import os
import torch.nn as nn
from torchsummary import summary
from deepWiseUNet import Network
from deepWiseUNetTwoInput import Network2
from twoInputUnet import Network3

def select_device(device='', apex=True, batch_size=None):
    cpu_request = device.lower() == 'cpu'
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device  # check availablity

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, 'batch-size %g not multiple of GPU count %g' % (batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = 'Using CUDA ' + ('Apex ' if apex else '')  # apex for mixed precision https://github.com/NVIDIA/apex
        for i in range(0, ng):
            if i == 1:
                s = ' ' * len(s)
            print("%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
                  (s, i, x[i].name, x[i].total_memory / c))
    else:
        print('Using CPU')

    print('')  # skip a line
    return torch.device('cuda:0' if cuda else 'cpu')

def main():
    parser = argparse.ArgumentParser(description='CNN Example')
    parser.add_argument('--rootDir', type=str, default='/infodev1/phi-data/shi/kneeX-ray/predictionBasedYoloOuput/512Sigma',
                        metavar='/infodev1/phi-data/shi/kneeX-ray/predictionBasedYoloOuput/512Sigma',
                        help='experiment data folder path')
    parser.add_argument('--batchSize', type=int, default=2, metavar='2',
                        help='input batch size for training (default: 2)')
    parser.add_argument('--epochs', type=int, default=48, metavar='48',
                        help='number of epochs to train (default:48)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='0.0001',
                        help='learning rate.')
    parser.add_argument('--fcNodes', type=int, default=1024)
    parser.add_argument('--resolution', default=512)
    parser.add_argument('--flag', default='hip')
    parser.add_argument('--sigma', type=int, default=20)
    parser.add_argument('--prediction', type=int, default=0)

    parser.add_argument('--originalUNet',type=bool, default=True)
    parser.add_argument('--deepWiseUNet', type=bool,default=False)
    parser.add_argument('--deepWiseUNetTwoInput',type=bool, default=False)
    parser.add_argument('--twoInputUNet', type=bool, default=False)
    parser.add_argument('--nin', default=1)
    parser.add_argument('--out', default=1)
    parser.add_argument('--nestCVTestIdx', type=int, default=1)
    parser.add_argument('--nestCVFoldIdx', type=int, default=1)
    parser.add_argument('--CV', type=bool, default=True)

    args = parser.parse_args()
    print(args)
    if args.flag == 'hip':
        deviceNum = '0'
    if args.flag == 'knee':
        deviceNum = '1'
    if args.flag == 'ankle':
        deviceNum = '2'
    device = select_device(deviceNum, batch_size=args.batchSize)

    if args.originalUNet == True:
        model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=1, out_channels=1,
                              init_features=32, pretrained=False)
    elif args.deepWiseUNet == True:
        model = Network(args)
    elif args.deepWiseUNetTwoInput == True:
        model = Network2(args)
    elif args.twoInputUNet == True:
        model = Network3(args)

    if torch.cuda.device_count() > 1:
        # model = nn.DataParallel(model, device_ids=[2, 1, 0])
        # model.to(f'cuda:{model.device_ids[0:2]}')
        model = nn.DataParallel(model)

    model.to(device)
    # summary(model, (1, 1024,1024),(1,512,512))
    total = sum([param.nelement() for param in model.parameters()])
    print('Number of parameters: ', total)

    trainer = TrainTestPrediction(model, device, args)
    trainer.train()


if __name__ == '__main__':
    main()
