import os, time
import shutil
import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F

def experiment_name_non_mnist(source='Amazon',
                             target='Webcam',
                             arch='Resnet50',
                             epochs=20,
                             rounds=15,
                             batch_size=32,
                             init_tgt_port=0.2,
                             max_tgt_port=0.5,
                             tgt_port_step=0.05,
                             lr=0.01,
                             add_name=''):
    exp_name = 'source_'+source+'_target_'+target
    exp_name += '_arch_'+str(arch)
    exp_name += '_rounds_' + str(rounds)
    exp_name += '_eph_' + str(epochs)
    exp_name +='_bs_'+str(batch_size)
    exp_name += '_init_tgt_port_' + str(init_tgt_port)
    exp_name += '_max_tgt_port_' + str(max_tgt_port)
    exp_name += '_tgt_port_step_' + str(tgt_port_step)
    # exp_name += '_init_src_port_' + str(init_src_port)
    # exp_name += '_min_src_port_' + str(min_src_port)
    # exp_name += '_src_port_step_' + str(src_port_step)
    exp_name += '_lr_' + str(lr)
    if add_name!='':
        exp_name += '_add_name_'+str(add_name)

    # exp_name += strftime("_%Y-%m-%d_%H:%M:%S", gmtime())
    print('experiement name: ' + exp_name)
    return exp_name


def copy_script_to_folder(caller_path, folder):
    script_filename = caller_path.split('/')[-1]
    script_relative_path = os.path.join(folder, script_filename)
    # Copying script
    shutil.copy(caller_path, script_relative_path)



def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()


def mixup_data(x, y, alpha):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def to_one_hot(inp, num_classes):
    y_onehot = torch.FloatTensor(inp.size(0), num_classes)
    y_onehot.zero_()

    y_onehot.scatter_(1, inp.unsqueeze(1).data.cpu(), 1)

    return Variable(y_onehot.cuda(), requires_grad=False)


def mixup_process(out, target_reweighted, lam):
    indices = np.random.permutation(out.size(0))
    out = out * lam + out[indices] * (1 - lam)
    target_shuffled_onehot = target_reweighted[indices]
    target_reweighted = target_reweighted * lam + target_shuffled_onehot * (1 - lam)

    # t1 = target.data.cpu().numpy()
    # t2 = target[indices].data.cpu().numpy()
    # print (np.sum(t1==t2))
    return out, target_reweighted

def get_lambda(alpha=1.0):
    '''Return lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    return lam

def adjust_learning_rate(optimizer, current_epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate
    assert len(args.gammas) == len(args.schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(args.gammas, args.schedule):
        if (current_epoch >= step):
            lr = lr * gamma
        else:
            break
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = lr * 10
    return lr

def convert_secs2time(epoch_time):
  need_hour = int(epoch_time / 3600)
  need_mins = int((epoch_time - 3600*need_hour) / 60)
  need_secs = int(epoch_time - 3600*need_hour - 60*need_mins)
  return need_hour, need_mins, need_secs

def time_string():
  ISOTIMEFORMAT='%Y-%m-%d %X'
  string = '[{}]'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
  return string


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


class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""

    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        assert total_epoch > 0
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]
        self.epoch_losses = self.epoch_losses - 1

        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]
        self.epoch_accuracy = self.epoch_accuracy

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        assert idx >= 0 and idx < self.total_epoch, 'total_epoch : {} , but update with the {} index'.format(
            self.total_epoch, idx)
        self.epoch_losses[idx, 0] = train_loss
        self.epoch_losses[idx, 1] = val_loss
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1
        return self.max_accuracy(False) == val_acc

    def max_accuracy(self, istrain):
        if self.current_epoch <= 0: return 0
        if istrain:
            return self.epoch_accuracy[:self.current_epoch, 0].max()
        else:
            return self.epoch_accuracy[:self.current_epoch, 1].max()

    def plot_curve(self, save_path):
        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1200, 800
        legend_fontsize = 10
        scale_distance = 48.8
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 5
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis * 50, color='g', linestyle=':', label='train-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis * 50, color='y', linestyle=':', label='valid-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print('---- save figure {} into {}'.format(title, save_path))
        plt.close(fig)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    _, target = torch.max(target,1)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(state, is_best, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)


def get_1x_lr_params_NOscale(model, args):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    b = []
    if args.arch == 'resnet50':
        b.append(model.conv1)
        b.append(model.bn1)
        b.append(model.layer1)
        b.append(model.layer2)
        b.append(model.layer3)
        b.append(model.layer4)
    else:
        b.append(model.features)

    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj += 1
                if k.requires_grad:
                    yield k

def get_10x_lr_params(model, args):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = []
    if args.arch == 'resnet50':
        b.append(model.fc)
    else:
        b.append(model.classifier)
        b.append(model.fc)
    print(b)

    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj += 1
                if k.requires_grad:
                    yield k



# def discard(model, log):
#     model.eval()
#     print_log('\n==>>Strating discard training sampples', log)
#     end = time.time()
#
#     f = open(s_list_path, 'r')
#     src_dirs = f.readlines()
#
#     s_loaderr = torch.utils.data.DataLoader(load_data.Office(s_list_path, training=False),
#                                              batch_size=1, num_workers=args.workers)
#     n_loaderr = torch.utils.data.DataLoader(load_data.Office(n_list_path, training=False),
#                                             batch_size=1, num_workers=args.workers)
#
#     for i, (input, target) in enumerate(s_loaderr):
#         if args.use_cuda:
#             target = target.cuda(async=True)
#             input = input.cuda()
#         with torch.no_grad():
#             input_var = Variable(input)
#             target_var = Variable(target)
#         hidden_ft = model(input_var, hidden_out=True)
#         fty = torch.cat([hidden_ft, torch.unsqueeze(target_var, 1)], 1)
#         if i == 0:
#             s_hidden_fts = fty
#         else:
#             s_hidden_fts = torch.cat([s_hidden_fts, fty], 0)
#
#     for i, (input, target) in enumerate(n_loaderr):
#         if args.use_cuda:
#             target = target.cuda(async=True)
#             input = input.cuda()
#         with torch.no_grad():
#             input_var = Variable(input)
#             target_var = Variable(target)
#         hidden_ft = model(input_var, hidden_out=True)
#         fty = torch.cat([hidden_ft, torch.unsqueeze(target_var, 1)], 1)
#         if i == 0:
#             hidden_fts = torch.cat([s_hidden_fts, fty], 0)
#         else:
#             hidden_fts = torch.cat([hidden_fts, fty], 0)
#
#     # compute centroids
#     centroids = torch.zeros(args.num_classes, 2048)
#     for i in range(args.num_classes):
#         centroids[i] = torch.mean(hidden_fts[torch.where(hidden_fts[:,-1]==i)],dim=1)[:-1]
#
#     src_cos = [[] for k in range(args.num_classes)]
#     src_cos_idx = [[] for k in range(args.num_classes)]
#     for i in range(len(s_loaderr)):
#         cla = s_hidden_fts[i,-1]
#         similarity = torch.cosine_similarity(centroids[cla], s_hidden_fts[i], dim=0)
#         similarity = similarity.item()
#         src_cos[cla].append(similarity)
#         src_cos_idx[cla].append((i, similarity))
#
#     for i in range(args.num_classes):
#         src_cos[i].sort()
#         len_cls = len(src_cos[i])
#         len_cls_thresh = int(math.floor(len_cls * args.src_dis_portion))
#         for j in src_cos_idx[i]:
#             if j[1] <= len_cls_thresh:
#                 src_dirs.pop(j[0])
#
#     print_log(
#         '[Time %.3f] [source images number: %03d]' % (time.time() - end, len(src_dirs)),
#         log)
#
#     f = open(n_list_path, 'a')
#     f.writelines(src_dirs)
#     f.close()



# def discard(model, log):
#     model.eval()
#     print_log('\n==>>Strating discard training sampples', log)
#     end = time.time()
#
#     f = open(s_list_path, 'r')
#     src_dirs = f.readlines()
#
#     s_loaderr = torch.utils.data.DataLoader(load_data.Office(s_list_path, training=False),
#                                             batch_size=1, num_workers=args.workers)
#
#     conf_dict = {k: [] for k in range(args.num_classes)}
#     index_dict = {k: [] for k in range(args.num_classes)}
#
#     for i, (input, target) in enumerate(s_loaderr):
#         if args.use_cuda:
#             target = target.cuda(async=True)
#             input = input.cuda()
#         with torch.no_grad():
#             input_var = Variable(input)
#             target_var = Variable(target)
#         # compute output
#         output, reweighted_target = model(input_var, target_var)
#         loss = bce_loss(softmax(output), reweighted_target)
#         loss, target = loss.item(), target.item()
#         conf_dict[target].append(loss)
#         index_dict[target].append((i, loss))
#
#     dis_dirs = []
#     for idx_cls in range(args.num_classes):
#         conf_dict[idx_cls].sort(reverse=True)
#         len_cls = len(conf_dict[idx_cls])
#         len_cls_thresh = int(math.floor(len_cls * args.src_dis_portion))
#         if len_cls_thresh != 0:
#             cls_thresh = conf_dict[idx_cls][len_cls_thresh - 1]
#             for i in index_dict[idx_cls]:
#                 if i[1] >= cls_thresh:
#                     dis_dirs.append(src_dirs[i[0]])
#
#     for dir in dis_dirs:
#         src_dirs.remove(dir)
#
#
#     print_log(
#         '[Time %.3f] [source images number: %03d]' % (time.time() - end, len(src_dirs)),
#         log)
#
#     f = open(n_list_path, 'a')
#     f.writelines(src_dirs)
#     f.close()


def softCrossEntropy(inputs, target, reduce=True):
    log_likelihood = -F.log_softmax(inputs, dim=1)
    sample_num, class_num = target.shape
    if reduce:
        loss = torch.sum(torch.mul(log_likelihood, target)) / sample_num
    else:
        loss = torch.sum(torch.mul(log_likelihood, target), dim=1)

    return loss


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]



class EMA():
    def __init__(self, decay, shape):
        self.decay = decay

    def register(self, val):
        self.shadow = val

    def get(self):
        return self.shadow

    def update(self, x):
        new_average = (1.0 - self.decay) * x + self.decay * self.shadow
        self.shadow = new_average


# class EMA:
#     def __init__(self, decay):
#         self.decay = decay
#         self.shadow = {}
#
#     def register(self, model):
#         for name, param in model.named_parameters():
#             if param.requires_grad:
#                 self.shadow[name] = param.data.clone()
#         self.params = self.shadow.keys()
#
#     def __call__(self, model):
#         if self.decay > 0:
#             for name, param in model.named_parameters():
#                 if name in self.params and param.requires_grad:
#                     self.shadow[name] -= (1 - self.decay) * (self.shadow[name] - param.data)
#                     param.data = self.shadow[name]


def copy_folder_to_folder(caller_path, folder):
    if not os.path.exists(folder):
        script_filename = caller_path.split('/')[-1]
        script_relative_path = os.path.join(folder, script_filename)
        # Copying script
        shutil.copytree(caller_path, script_relative_path)

def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = 1.0
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


class ECELoss():
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, softmaxes, labels):
        confidences, predictions = torch.max(softmaxes, 1)
        labels = torch.argmax(labels, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=softmaxes.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece