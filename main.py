from models.resnet import resnet
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from utils import experiment_name_non_mnist, copy_script_to_folder, print_log, adjust_learning_rate, convert_secs2time, \
    copy_folder_to_folder, \
    time_string, AverageMeter, accuracy, RecorderMeter, save_checkpoint, get_1x_lr_params_NOscale, get_10x_lr_params, \
    softCrossEntropy, interleave, EMA
import os, time, shutil, math
from load_data import return_dataset, return_psu_dataset
import sys

parser = argparse.ArgumentParser(description='UDA with manifold mixup regularization', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--source', type=str, default='amazon', choices=['Art', 'Clipart', 'Product', 'Real', 'amazon', 'webcam', 'dslr', 'c', 'i', 'p', 'amazon10', 'webcam10', 'dslr10', 'caltech10'], help='Choose soure dataset.')
parser.add_argument('--target', type=str, default='webcam', choices=['Art', 'Clipart', 'Product', 'Real', 'amazon', 'webcam', 'dslr', 'c', 'i', 'p', 'amazon10', 'webcam10', 'dslr10', 'caltech10'], help='Choose target dataset.')
parser.add_argument('--root_dir', type = str, default = './experiments/', help='folder where results are to be stored')
parser.add_argument('--arch',  type=str, default='resnet50')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs per round.')
parser.add_argument('--rounds', type=int, default=20, help='Number of rounds to train.')
parser.add_argument('--num_classes', type=int, default=31, help='training data categories')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', type=float, default=0.0005, help='Weight decay (L2 penalty).')
parser.add_argument('--schedule', type=int, nargs='+', default=[20], help='Decrease learning rate at these rounds.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1], help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
parser.add_argument('--print_freq', default=20, type=int, metavar='N', help='print frequency (default: 200)')
parser.add_argument('--dropout', action='store_true', default=False,
                    help='whether to use dropout or not in final layer')
parser.add_argument('--init-tgt-port', default=0.1, type=float, dest='init_tgt_port',
                    help='The initial portion of target to determine kc')
parser.add_argument('--max-tgt-port', default=0.9, type=float, dest='max_tgt_port',
                    help='The max portion of target to determine kc')
parser.add_argument('--tgt-port-step', default=0.05, type=float, dest='tgt_port_step',
                    help='The portion step in target domain in every round of self-paced self-trained neural network')
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 2)')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--T', default=0.5, type=float)
parser.add_argument('--EMA_momentum', default=0.95, type=float)
parser.add_argument('--dataset', type=str, default='office',
                    choices=['image-clef', 'office', 'office_home', 'office_caltech'],
                    help='the name of dataset')
parser.add_argument('--pretrained', type=bool, default=True)
parser.add_argument('--pretrained_checkpoint', type=bool, default=False)
parser.add_argument('--gpu_id', type=int, default=0,  help='gpu id')
parser.add_argument('--run_id', type=int, default=0,  help='run id')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
args.use_cuda = args.ngpu>0 and torch.cuda.is_available()

out_str = str(args)
print(out_str)

exp_name = experiment_name_non_mnist(source=args.source,
                                     target=args.target,
                                     arch=args.arch,
                                     epochs=args.epochs,
                                     rounds=args.rounds,
                                     batch_size=args.batch_size,
                                     init_tgt_port=args.init_tgt_port,
                                     max_tgt_port=args.max_tgt_port,
                                     tgt_port_step=args.tgt_port_step,
                                     lr=args.learning_rate,
                                     add_name=str(args.run_id)
                                     )

exp_dir = args.root_dir + exp_name
args.exp_dir = os.path.join(args.root_dir, exp_name)
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

pseudo_label_acc = []
pseudo_label_num = []
network_acc = []
best_acc = 0
cudnn.benchmark = True
src_cls_prob = np.ones((1, args.num_classes)) / args.num_classes

base_path = args.exp_dir + '/data_list/'
trg_img_pth_file_unl = os.path.join(base_path, args.target + '_img.txt')
trg_label_pth_file_unl = os.path.join(base_path, args.target + '_label.txt')
trg_img_pth_file_psu = os.path.join(base_path,
                                    'pesudo_target_images_' + args.target + '_img.txt')
trg_label_pth_file_psu = os.path.join(base_path,
                                      'pesudo_target_images_' + args.target + '_label.txt')


def train(source_loader, target_loader, model, optimizer, epoch, args, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    source_train_loader = iter(source_loader)
    target_train_loader = iter(target_loader)

    steps = max(len(source_loader), len(target_loader))

    for i in range(steps):
        try:
            xs, ys = source_train_loader.next()
        except:
            source_train_loader = iter(source_loader)
            xs, ys = source_train_loader.next()

        try:
            xt, yt = target_train_loader.next()
        except:
            target_train_loader = iter(target_loader)
            xt, yt = target_train_loader.next()
        ys = ys.float().cuda()
        yt = yt.float().cuda()
        xs = xs.cuda()
        xt = xt.cuda()

        # mixup
        all_inputs = torch.cat([xs, xt], dim=0)
        all_targets = torch.cat([ys, yt], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        input_a = list(torch.split(input_a, args.batch_size))
        input_a = interleave(input_a, args.batch_size)

        logits = [model(input_a[0]), model(input_a[1])]
        logits = interleave(logits, args.batch_size)

        logits_a = logits[0]
        logits_b = logits[1]

        Lx = softCrossEntropy(logits_a, target_a[:args.batch_size])
        Lu = softCrossEntropy(logits_b, target_a[args.batch_size:])
        loss = Lx + Lu

        outputs = torch.cat(logits, 0)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs, all_targets, topk=(1, 5))
        losses.update(loss.item(), args.batch_size)
        top1.update(prec1.item(), args.batch_size)
        top5.update(prec5.item(), args.batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                      'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                epoch, i, steps, batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5) + time_string(), log)

    print_log(
        '  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5,
                                                                                              error1=100 - top1.avg),
        log)
    return top1.avg, top5.avg, losses.avg


def validate(target_loader_unl, model, ema, current_epoch, log):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    l_loader_iter = iter(target_loader_unl)

    with torch.no_grad():
        for i in range(len(target_loader_unl)):
            (input1, input2), target = l_loader_iter.next()
            input1 = input1.cuda()
            input2 = input2.cuda()
            target = target.float()
            target = target.cuda()
            output1 = model(input1)
            output2 = model(input2)
            loss1 = softCrossEntropy(output1, target)
            loss2 = softCrossEntropy(output2, target)
            output1 = torch.softmax(output1, dim=1)
            output2 = torch.softmax(output2, dim=1)
            loss = (loss1 + loss2) / 2.0

            # measure accuracy and record loss
            prec1, prec5 = accuracy((output1 + output2).data, target, topk=(1, 5))
            losses.update(loss.item(), input1.size(0))
            top1.update(prec1.item(), input1.size(0))
            top5.update(prec5.item(), input1.size(0))

            if i == 0:
                logits1 = output1.cpu().data
                logits2 = output2.cpu().data
            else:
                logits1 = np.concatenate((logits1, output1.cpu().data), 0)
                logits2 = np.concatenate((logits2, output2.cpu().data), 0)

    pred_cls_prob = (np.mean(logits1, axis=0, keepdims=True) + np.mean(logits2, axis=0, keepdims=True)) / 2.0
    # print_log('prediction class probability', log)
    # print_log(str(pred_cls_prob), log)
    logits1 = logits1 * src_cls_prob / pred_cls_prob
    logits1 = logits1 / np.sum(logits1, axis=1, keepdims=True)
    logits2 = logits2 * src_cls_prob / pred_cls_prob
    logits2 = logits2 / np.sum(logits2, axis=1, keepdims=True)
    logits = ((logits1 + logits2) / 2.0) ** (1 / args.T)
    logits = logits / logits.sum(axis=1, keepdims=True)

    if current_epoch == args.epochs-5:
        ema.register(logits)
    if current_epoch > args.epochs-5:
        ema.update(logits)

    print_log(
        '  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f} Loss: {losses.avg:.3f} '.format(
            top1=top1, top5=top5, error1=100 - top1.avg, losses=losses), log)

    return top1.avg, losses.avg


def label(ema, log):
    print_log('\n==>>Strating pesudo label', log)
    # switch to evaluate mode
    conf_dict = {k: [] for k in range(args.num_classes)}
    f = open(trg_img_pth_file_unl, 'r')
    unlabel_target_images = f.readlines()
    unlabel_target_labels = np.loadtxt(trg_label_pth_file_unl)
    pesudo_target_images = []
    pesudo_target_labels = []
    true_target_labels = []

    pred_prob_mod = ema.get()

    for i in range(pred_prob_mod.shape[0]):
        value, ind = np.max(pred_prob_mod[i]), np.argmax(pred_prob_mod[i])
        conf_dict[ind].append(value)

    cls_thresh = np.ones(args.num_classes, dtype=np.float32)
    for idx_cls in range(args.num_classes):
        if conf_dict[idx_cls] != None:
            conf_dict[idx_cls].sort(reverse=True)
            len_cls = len(conf_dict[idx_cls])
            len_cls_thresh = int(math.floor(len_cls * args.tgt_portion))
            if len_cls_thresh != 0:
                cls_thresh[idx_cls] = conf_dict[idx_cls][len_cls_thresh - 1]

    weighted_prob = pred_prob_mod / cls_thresh
    weighted_conf = np.amax(weighted_prob, axis=1)
    for idx, pth in enumerate(unlabel_target_images):
        if weighted_conf[idx] >= 1.0:
            pesudo_target_images.append(pth)
            pesudo_target_labels.append(pred_prob_mod[idx])
            true_target_labels.append(unlabel_target_labels[idx])

    pesudo_target_labels = np.array(pesudo_target_labels)
    true_target_labels = np.array(true_target_labels)
    f = open(trg_img_pth_file_psu, 'w')
    f.writelines(pesudo_target_images)
    np.savetxt(trg_label_pth_file_psu, pesudo_target_labels, fmt='%1.3f')
    acc = np.mean(np.sum(pesudo_target_labels * true_target_labels, axis=1), axis=0)

    pseudo_label_acc.append(acc)
    print_log('     pseudo label accuracy:%s' % acc, log)

    print_log('==>>finish pesudo label', log)


def main():
    global best_acc

    copy_script_to_folder(os.path.abspath(__file__), exp_dir)
    copy_folder_to_folder('./data_list/', args.exp_dir + '/data_list/')
    result_png_path = os.path.join(exp_dir, 'results.png')

    log = open(os.path.join(exp_dir, 'log.txt'.format(args.manualSeed)), 'w')
    print_log('save path : {}'.format(exp_dir), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(torch.backends.cudnn.version()), log)

    source_dataset, target_dataset_unl = return_dataset(args)
    targets_nums = len(target_dataset_unl)
    source_loader = \
        torch.utils.data.DataLoader(source_dataset,
                                    batch_size=args.batch_size,
                                    num_workers=args.workers,
                                    shuffle=True, drop_last=True)
    target_loader = source_loader
    target_loader_unl = \
        torch.utils.data.DataLoader(target_dataset_unl,
                                    batch_size=args.batch_size, num_workers=args.workers,
                                    shuffle=False, drop_last=False)

    print_log("=> creating model '{}'".format(args.arch), log)
    net = resnet(args)
    print_log("=> network :\n {}".format(net), log)
    net = net.cuda()

    print_log("=> Loaded pretrained checkpoint for {} model".format(args.arch), log)

    optimizer = torch.optim.SGD([{'params': get_1x_lr_params_NOscale(net, args), 'lr': args.learning_rate},
                                 {'params': get_10x_lr_params(net, args), 'lr': 10 * args.learning_rate}],
                                lr=args.learning_rate, momentum=args.momentum, weight_decay=args.decay)

    recorder = RecorderMeter(args.epochs * args.rounds)

    # Main loop
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    ema = EMA(decay=args.EMA_momentum, shape=(targets_nums, args.num_classes))

    epoch_time = AverageMeter()
    start_time = time.time()

    for round in range(0, args.rounds):
        args.tgt_portion = min(args.init_tgt_port + round * args.tgt_port_step, args.max_tgt_port)

        # training
        for epoch in range(0, args.epochs):
            current_epoch = round * args.epochs + epoch
            current_learning_rate = adjust_learning_rate(optimizer, current_epoch, args)
            total_epoch = args.rounds * args.epochs
            need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (total_epoch - current_epoch))
            need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

            print_log('\n==>>{:s} [round={:02d}/{:02d}] [Epoch={:02d}/{:02d}] {:s} [learning_rate={:8.6f}]'.format(
                time_string(), round,
                args.rounds, epoch, args.epochs, need_time,
                current_learning_rate) \
                      + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False),
                                                                         100 - recorder.max_accuracy(False)), log)
            # train for one epoch
            tr_acc, tr_acc5, tr_los = train(source_loader, target_loader, net, optimizer, epoch, args, log)

            # evaluate on validation set
            val_acc, val_los = validate(target_loader_unl, net, ema, current_epoch, log)
            network_acc.append(val_acc)

            train_loss.append(tr_los)
            train_acc.append(tr_acc)
            test_loss.append(val_los)
            test_acc.append(val_acc)

            dummy = recorder.update(current_epoch, tr_los, tr_acc, val_los, val_acc)

            is_best = False
            if val_acc > best_acc:
                is_best = True
                best_acc = val_acc

            save_checkpoint({
                'round': round,
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': net.state_dict(),
                'recorder': recorder,
                'optimizer': optimizer.state_dict(),
            }, is_best, exp_dir, 'checkpoint.pth.tar')

            # measure elapsed time
            epoch_time.update(time.time() - start_time)
            start_time = time.time()
            recorder.plot_curve(result_png_path)

        # pesudo labels
        label(ema, log)
        target_dataset_psu = return_psu_dataset(args)
        pseudo_label_num.append(len(target_dataset_psu))
        target_loader = torch.utils.data.DataLoader(target_dataset_psu,
                                                    batch_size=args.batch_size, shuffle=True, drop_last=True,
                                                    num_workers=args.workers)

    print_log(str(pseudo_label_acc), log)
    print_log(str(pseudo_label_num), log)
    print_log(str(network_acc), log)


if __name__ == '__main__':
    main()