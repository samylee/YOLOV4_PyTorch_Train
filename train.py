import torch
from torch.utils.data import DataLoader

from data.dataset import VOCDataset
from models.YOLOV4 import YOLOV4
from utils.loss import YOLOV4Loss
from utils import transforms
from utils.utils import load_darknet_pretrain_weights


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


if __name__ == '__main__':
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    base_lr = 1e-4
    weight_decay = 0

    lr_steps = [200, 250]
    num_epochs = 280
    batch_size = 32
    B, C = 3, 20
    net_size = 416
    net_random = 0 # for random net_size
    anchors = [[10,14],  [23,27],  [37,58],  [81,82],  [135,169], [344,319]]
    masks = [[3, 4, 5], [1, 2, 3]]

    pretrain = 'pretrain/yolov4-tiny.weights'
    train_label_list = 'data/voc0712/train.txt'

    print_freq = 5
    save_freq = 5

    # def model
    yolov4 = YOLOV4(B=B, C=C)
    load_darknet_pretrain_weights(yolov4, pretrain)
    yolov4.to(device)

    # def loss
    criterion = YOLOV4Loss(B, C, anchors=anchors, masks=masks, device=device)

    # def optimizer
    optimizer = torch.optim.Adam(yolov4.parameters(), lr=base_lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_steps)

    # def dataset
    transforms = transforms.Compose([
        transforms.RandomCrop(jitter=0.3, resize=1.5, net_size=net_size),
        transforms.RandomFlip(prob=0.5),
        transforms.RandomHue(hue=0.1, prob=0.5),
        transforms.RandomSaturation(sat=1.5, prob=0.5),
        transforms.RandomExposure(exp=1.5, prob=0.5)
    ])
    train_dataset = VOCDataset(train_label_list, transform=transforms, is_train=True, net_size=net_size, net_random=net_random)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=train_dataset.collate_fn)

    print('Number of training images: ', len(train_dataset))

    # train
    for epoch in range(num_epochs):
        yolov4.train()
        total_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            current_lr = get_lr(optimizer)

            inputs = inputs.to(device)
            targets = targets.to(device)

            preds = yolov4(inputs)

            loss = 0.
            for idx, pred in enumerate(preds):
                loss += criterion(pred, targets, idx, inputs.size(2))
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print current loss.
            if i % print_freq == 0:
                print('Epoch [%d/%d], Iter [%d/%d], LR: %.6f, Size: %d, Loss: %.4f, Average Loss: %.4f'
                      % (epoch, num_epochs, i, len(train_loader), current_lr, inputs.shape[2], loss.item(), total_loss / (i+1)))

        lr_scheduler.step()
        if epoch % save_freq == 0:
            torch.save(yolov4.state_dict(), 'weights/yolov4_' + str(epoch) + '.pth')

    torch.save(yolov4.state_dict(), 'weights/yolov4_final.pth')