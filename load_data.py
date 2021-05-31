import os
from torchvision import transforms
from loaders.data_list import Imagelists_VISDA, return_classlist, Imagelists_VISDA_twice
import  torch


class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2



def return_dataset(args):
    base_path = args.exp_dir+'/data_list'
    if args.dataset == 'office':
        root = ''
    else:
        root = '../dataset/%s' % args.dataset

    src_img_pth_file = os.path.join(base_path, args.source + '_img.txt')
    src_label_pth_file = os.path.join(base_path, args.source + '_label.txt')
    trg_img_pth_file_unl = os.path.join(base_path, args.target + '_img.txt')
    trg_label_pth_file_unl = os.path.join(base_path, args.target + '_label.txt')


    crop_size = 224
    data_transforms = {
        'train': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    source_dataset = Imagelists_VISDA(src_img_pth_file, src_label_pth_file, root=root,
                                      transform=data_transforms['train'])
    target_dataset_unl = Imagelists_VISDA_twice(trg_img_pth_file_unl, trg_label_pth_file_unl, root=root,
                                          transform=TransformTwice(data_transforms['train']))


    return source_dataset, target_dataset_unl


def return_psu_dataset(args):
    base_path = args.exp_dir+'/data_list'
    if args.dataset == 'office':
        root = ''
    else:
        root = '../dataset/%s'%args.dataset

    trg_img_pth_file_psu = os.path.join(base_path,
                                        'pesudo_target_images_' + args.target + '_img.txt')
    trg_label_pth_file_psu = os.path.join(base_path,
                                          'pesudo_target_images_' + args.target + '_label.txt')


    crop_size = 224

    data_transforms = {
        'train': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    target_dataset_psu = Imagelists_VISDA(trg_img_pth_file_psu, trg_label_pth_file_psu, root=root,
                                      transform=data_transforms['train'])

    return  target_dataset_psu


def retrun_test_dataset(args):
    base_path = args.exp_dir+'/data_list'
    if args.dataset == 'office':
        root = ''
    else:
        root = '../dataset/%s' % args.dataset

    src_img_pth_file = os.path.join(base_path, args.source + '_img.txt')
    src_label_pth_file = os.path.join(base_path, args.source + '_label.txt')
    trg_img_pth_file_unl = os.path.join(base_path, args.target + '_img.txt')
    trg_label_pth_file_unl = os.path.join(base_path, args.target + '_label.txt')

    crop_size = 224

    data_transforms = {
        'train': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    source_dataset = Imagelists_VISDA(src_img_pth_file, src_label_pth_file, root=root,
                                      transform=data_transforms['test'])
    target_dataset_unl = Imagelists_VISDA(trg_img_pth_file_unl, trg_label_pth_file_unl, root=root,
                                                transform=data_transforms['test'])

    return source_dataset, target_dataset_unl


def per_image_standardization(x):
    y = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
    mean = y.mean(dim=1, keepdim = True).expand_as(y)
    std = y.std(dim=1, keepdim = True).expand_as(y)
    adjusted_std = torch.max(std, 1.0/torch.sqrt(torch.cuda.FloatTensor([x.shape[1]*x.shape[2]*x.shape[3]])))
    y = (y- mean)/ adjusted_std
    standarized_input =  y.view(x.shape[0],x.shape[1],x.shape[2],x.shape[3])
    return standarized_input