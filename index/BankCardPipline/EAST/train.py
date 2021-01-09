import argparse
import torch
from .data_loader import MyDataset
from .model import EastModel
from tensorboardX import SummaryWriter
import os

from .helper import to_cuda, loss_function, collate_fn

import warnings  
warnings.filterwarnings("ignore")  

def main(opt):
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)
    writer = SummaryWriter('log2')
    use_cuda = torch.cuda.is_available()
    train_dataset = MyDataset(opt.train_data, opt.input_size)
    train_length=int(0.9* len(train_dataset))
    val_length=len(train_dataset)-train_length
    train_dataset,val_dataset=torch.utils.data.random_split(train_dataset,(train_length,val_length))
    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=opt.batch_size, 
                                            shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
    #                                         batch_size=8,
    #                                         shuffle=False)
    model = EastModel(opt.text_scale, opt.input_size)
    model = torch.nn.DataParallel(model)
    if use_cuda:
        model = model.cuda()
    model = model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-5)
    if opt.continue_model != '':
        print(f'loading pretrained model from {opt.continue_model}')
        model.load_state_dict(torch.load(opt.continue_model))
    global_batch = 0
    min_val_loss = float('inf')
    for epoch in range(opt.num_iter):
        for i,(imgs, score_maps, geo_maps) in enumerate(train_loader):
            if use_cuda:
                imgs, score_maps, geo_maps = map(to_cuda, (imgs,score_maps, geo_maps))
            F_score, F_geometry = model(imgs)
            loss = loss_function(score_maps, F_score, geo_maps, F_geometry)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_batch += 1
            
            if global_batch % opt.log_interval == 0:
                print("global batch:%d, train loss:%f"%(global_batch,loss.item()))
                writer.add_scalar('Loss/train_loss', loss.item(), global_batch)

            if global_batch % opt.save_interval == 0:
                torch.save(model.state_dict(), os.path.join(opt.save_path,'model_{}.pkl'.format(epoch)))
                model = model.eval()
                losses = []
                for i,(imgs, score_maps, geo_maps) in enumerate(val_loader):
                    if use_cuda:
                        imgs, score_maps, geo_maps = map(to_cuda, (imgs,score_maps, geo_maps))
                    F_score, F_geometry = model(imgs)
                    loss = loss_function(score_maps, F_score, geo_maps, F_geometry)
                    losses.append(loss.item())
                loss = sum(losses) / len(losses)
                print("global batch:%d, valid loss:%f"%(global_batch,loss))
                writer.add_scalar('Loss/valid_loss', loss, global_batch)
                if loss < min_val_loss:
                    min_val_loss = loss
                    torch.save(model.state_dict(), os.path.join(opt.save_path,'model_{}.pkl'.format(epoch)))
                model = model.train()
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', help='Where to store logs and models')
    parser.add_argument('--train_data', required=True, help='path to training dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
    parser.add_argument('--num_iter', type=int, default=3000, help='number of iterations to train for')
    parser.add_argument('--continue_model', default='', help="path to model to continue training")
    parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
    parser.add_argument('--input_size', type=int, default=512, help='input image size')
    parser.add_argument('--text_scale', type=int, default=512, help='text scale')
    parser.add_argument('--log_interval', type=int, default=5, help='Interval between each log')
    parser.add_argument('--save_interval', type=int, default=500, help='Interval between each model save and val')
    parser.add_argument('--save_path', type=str, default="models", help='Interval between each model save')

    opt = parser.parse_args()

    main(opt)
