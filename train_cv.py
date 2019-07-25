import torch
import torch.autograd as autograd
import torch.nn.functional as F
from torch.nn.parallel.data_parallel import data_parallel

from dataset import *
from utils import *
from models import *
import tqdm

criterion = nn.CrossEntropyLoss().cuda()


def train(train_loader, val_loader, model, optimizer, args, model_path):
    model.cuda()

    steps = 0
    best_acc = 0
    best_loss = float('inf')

    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)

    train_info = {'epoch': [], 'train_loss': [], 'val_loss': [], 'metric': [], 'best': [], 'test_loss': [],
                  'test_acc': []}

    print(
        'epoch |   lr    |    %        |  loss  |  avg   |val loss| top1  |  top3   |  best  |test loss| top1  | time | save |')
    bg = time.time()
    train_iter = 0
    model.train()

    for epoch in range(1, args.epochs + 1):
        losses = []
        train_loss = 0
        last_val_iter = 0
        current_lr = get_lrs(optimizer)
        for batch_idx, batch in enumerate(train_loader):
            train_iter += 1
            feature, target = batch[0], batch[1]
            # feature.data.t_(), target.data.sub_(1)  # batch first, index align
            feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = data_parallel(model, feature)

            loss = criterion(logit, target)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

            train_loss += loss.item()
            losses.append(loss.item())

            print('\r {:4d} | {:.5f} | {:4d}/{} | {:.4f} | {:.4f} |'.format(
                epoch, float(current_lr[0]), args.batch_size * (batch_idx + 1), train_loader.num, loss.item(),
                                             train_loss / (train_iter - last_val_iter)), end='')

            if train_iter > 0 and train_iter % args.iter_val == 0:

                top_1, top_3, val_loss, size = validate(val_loader, model)
                # test_top_1, tst_top_3, test_loss, _ = validate(test_loader, model)
                _save_ckp = ' '

                if val_loss < best_loss:
                    best_acc = top_1
                    best_loss = val_loss
                    save_checkpoint(model_path, model, optimizer)
                    _save_ckp = '*'

                print(' {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.2f} | {:4s} |'.format(val_loss, top_1, top_3, best_acc,
                                                                                     (time.time() - bg) / 60,
                                                                                     _save_ckp))

                train_info['epoch'].append(args.batch_size * (batch_idx + 1) / train_loader.num + epoch)
                train_info['train_loss'].append(train_loss / (batch_idx + 1))
                train_info['val_loss'].append(val_loss)
                train_info['metric'].append(top_1)
                train_info['best'].append(best_acc)

                log_df = pd.DataFrame(train_info)
                log_df.to_csv(model_path + '.csv')

                train_loss = 0
                last_val_iter = train_iter

                model.train()

    log_df = pd.DataFrame(train_info)
    log_df.to_csv(model_path + '.csv')
    print("Best accuracy is {:.4f}".format(best_acc))


def validate(data_loader, model):
    model.eval()
    corrects = []
    losses = []
    for batch in data_loader:
        feature, target = batch[0], batch[1]
        # feature.data.t_(), target.data.sub_(1)  # batch first, index align
        feature, target = feature.cuda(), target.cuda()

        with torch.no_grad():
            logit = model(feature)
            loss = criterion(logit, target)

            losses.append(loss.item())
            correct = metric(logit, target)
            corrects.append(correct.data.cpu().numpy())

    correct = np.concatenate(corrects)
    correct = correct.mean(0)
    loss = np.mean(losses)
    top = [correct[0], correct[0] + correct[1], correct[0] + correct[1] + correct[2]]
    size = len(data_loader.dataset)
    return top[0], top[2], loss, size


if __name__ == '__main__':
    args = argparser()
    args.vocab_size = 21
    print(args)

    EMBED_MODEL = ['PepCNNDeepEmbed']
    POS_MODEL = ['Transformer']
    if args.model in EMBED_MODEL:
        args.use_embed = True
    else:
        args.use_embed = False
    if args.model in POS_MODEL:
        args.position = True
    else:
        args.position = False

    assert os.path.isfile(args.data)
    assert os.path.isdir(args.checkpoint_path)

    parent_path = os.path.dirname(args.data)
    file_name = os.path.basename(args.data)
    cv_idx = file_name.find('cv')
    dfs = []
    for i in range(args.kfold):
        df_path = os.path.join(parent_path, file_name[:cv_idx] + "cv{}.txt".format(i))
        df = pd.read_csv(df_path, sep='\t', header=None)
        dfs.append(df)

    for i in range(args.kfold):
        df_train = [df for idx, df in enumerate(dfs) if idx != i]
        df_train = pd.concat(df_train)
        df_val = dfs[i]
        val_size = len(df_val)
        # df_val = df_val[:int(0.1 * val_size)]

        df_test = pd.read_csv(args.test_file, sep='\t', header=None)

        train_data = PepseqDataset(file_path=args.train_file)
        # test_data = PepseqDataset(file_path="/home/dunan/Documents/DeepFam_data/GPCR/cv_1/test.txt")
        test_data = PepseqDataset(file_path=args.test_file)

        train_loader = data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
        test_loader = data.DataLoader(test_data, batch_size=args.batch_size, num_workers=4)

        model = PepCNN()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.l2)

        model_path = os.path.join(args.checkpoint_path, 'model_cv{}.pth'.format(i))

        train(train_loader, test_loader, model, optimizer, args, model_path)