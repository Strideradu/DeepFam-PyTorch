import torch
import torch.nn.functional as F
from dataset import *
from utils import *
from models import PepCNN
import tqdm


def predict(args):
    model = PepCNN(num_class=args.num_classes)
    load_checkpoint(args.checkpoint_path, model)
    model.eval()

    predict_data = PepseqDataset(args.test_file)
    data_loader = data.DataLoader(predict_data, batch_size=args.batch_size)

    corrects = 0
    for batch in tqdm.tqdm(data_loader):
        feature, target = batch[0], batch[1]
        # feature.data.t_(), target.data.sub_(1)  # batch first, index align
        feature, target = feature.cuda(), target.cuda()

        logit = model(feature)

        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()
        print(torch.topk(logit, args.topk))

    size = len(data_loader.dataset)
    accuracy = 100 * corrects.data.cpu().numpy() / size


if __name__ == '__main__':
    args = argparser()
    predict(args)
