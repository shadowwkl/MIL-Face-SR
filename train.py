import os
import argparse
import shutil

import torch
from torch.backends import cudnn

from data import make_dataset
from utils import make_logger, list_dir_recursively_with_ignore, copy_files_and_create_dirs
from models.GAN import SRGAN
import pdb

# Load fewer layers of pre-trained models if possible
import torch.utils.data as data

from PIL import Image
import os
import os.path
from torch.utils.data import DataLoader
from torchvision import transforms

def default_loader(path):
    return Image.open(path).convert('RGB')

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath = line.strip()
            imlist.append(impath)

    return imlist

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = sorted(make_dataset(root))
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)


class ImageFilelist(data.Dataset):
    def __init__(self, flist, transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        # self.root = root
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        impath = self.imlist[index]
        img = self.loader(os.path.join('./', impath))
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.imlist)

def load(model, cpk_file):
    pretrained_dict = torch.load(cpk_file)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


def get_data_loader_list(input_folder, batch_size):
    transform_list = [
            # transforms.RandomHorizontalFlip(),
            transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
    transform = transforms.Compose(transform_list)
    dataset = ImageFilelist(input_folder, transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=1)
    return loader


def get_data_loader_folder(input_folder, batch_size):
    transform_list = [
            # transforms.RandomHorizontalFlip(),
            transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
    transform = transforms.Compose(transform_list)
    dataset = ImageFolder(input_folder, transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=1)
    return loader

def get_mask_loader_folder(input_folder, batch_size):
    transform_list = [
            # transforms.RandomHorizontalFlip(),
            # transforms.Resize(256),
            transforms.ToTensor()
            # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ]
    transform = transforms.Compose(transform_list)
    dataset = ImageFolder(input_folder, transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=1)
    return loader
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="StyleGAN pytorch implementation.")
    parser.add_argument('--config', default='./configs/encoder_Pixelwave_PSLoss_fix_v2_vgg_bicubic_ori_x2gen_128_nossim_V2_5GANLoss_testt.yaml')

    parser.add_argument("--start_depth", action="store", type=int, default=0,
                        help="Starting depth for training the network")
    parser.add_argument("--num_examplar", action="store", type=int, default=3,
                        help="num of examplar")
    parser.add_argument("--training_mode", type=bool, default=False,
                        help="Starting depth for training the network")

    parser.add_argument("--test_mode", type=bool, default=False,
                        help="Starting depth for training the network")

    parser.add_argument("--visualizePWAVE", action="store", type=bool, default=False,
                        help="Starting depth for training the network")

    parser.add_argument("--generator_file", action="store", type=str, default=None,
                        help="pretrained Generator file (compatible with my code)")
    parser.add_argument("--gen_shadow_file", action="store", type=str, default=None,
                        help="pretrained gen_shadow file")
    parser.add_argument("--discriminator_file", action="store", type=str, default=None,
                        help="pretrained Discriminator file (compatible with my code)")
    parser.add_argument("--gen_optim_file", action="store", type=str, default=None,
                        help="saved state of generator optimizer")
    parser.add_argument("--dis_optim_file", action="store", type=str, default=None,
                        help="saved_state of discriminator optimizer")
    parser.add_argument("--x2generator_file", action="store", type=str, default=None,
                        help="saved_state of discriminator optimizer--")

    parser.add_argument("--x2gen_optim_file", action="store", type=str, default=None,
                        help="saved_state of discriminator optimizer---")

    args = parser.parse_args()

    from config import cfg as opt
    # pdb.set_trace()

    opt.merge_from_file(args.config)
    opt.freeze()

    # make output dir
    output_dir = opt.output_dir
    # if os.path.exists(output_dir):
    #     raise KeyError("Existing path: ", output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # copy codes and config file
    files = list_dir_recursively_with_ignore('./', ignores=['diagrams', 'configs'])
    files = [(f[0], os.path.join(output_dir, "src", f[1])) for f in files]
    # pdb.set_trace()

    copy_files_and_create_dirs(files)
    shutil.copy2(args.config, output_dir)



    # logger
    logger = make_logger("project", opt.output_dir, 'testing')

    # pdb.set_trace()

    if args.training_mode:
        logger = make_logger("project", opt.output_dir, 'log')

    # device
    if opt.device == 'cuda':
        # os.environ['CUDA_VISIBLE_DEVICES'] = opt.device_id
        # num_gpus = len(opt.device_id.split(','))
        # logger.info("Using {} GPUs.".format(num_gpus))
        logger.info("Training on {}.\n".format(torch.cuda.get_device_name(0)))
        cudnn.benchmark = True
    device = torch.device(opt.device)


    ############ create the dataset for training ############
    # prepare your data path in a text file
    # The first one is the lr image, it can be hr, then the code will make it lr
    # e.g.
    #/xx/xx/xx/0.jpg
    #/xx/xx/xx/1.jpg
    #/xx/xx/xx/2.jpg
    #/xx/xx/xx/3.jpg

    dataset = get_data_loader_list('./data/celeba_128_list.txt',1)

    # pdb.set_trace()

    # init the network
    gan = SRGAN(
                         resolution=opt.dataset.resolution,
                         num_channels=opt.dataset.channels,
                         g_args=opt.model.gen,
                         d_args=opt.model.dis,
                         g_opt_args=opt.model.g_optim,
                         d_opt_args=opt.model.d_optim,
                         d_repeats=opt.d_repeats,
                         device=device
                )

    # Resume training from checkpoints

    # pdb.set_trace()
    start_epoch = 0
    if args.generator_file is not None:
        logger.info("Loading generator from: %s", args.generator_file)
        # style_gan.gen.load_state_dict(torch.load(args.generator_file))
        # Load fewer layers of pre-trained models if possible
        load(gan.gen, args.generator_file)
        load(gan.x2gen, args.x2generator_file)
    else:
        logger.info("Training from scratch...")

    if args.discriminator_file is not None:
        logger.info("Loading discriminator from: %s", args.discriminator_file)
        gan.dis.load_state_dict(torch.load(args.discriminator_file))

    if args.gen_optim_file is not None:
        logger.info("Loading generator optimizer from: %s", args.gen_optim_file)
        gan.gen_optim.load_state_dict(torch.load(args.gen_optim_file))
        gan.x2gen_optim.load_state_dict(torch.load(args.x2gen_optim_file))
        
    if args.dis_optim_file is not None:
        logger.info("Loading discriminator optimizer from: %s", args.dis_optim_file)
        gan.dis_optim.load_state_dict(torch.load(args.dis_optim_file))
        # start_epoch = int(args.dis_optim_file.split('/')[-1].split('_')[-2])
        # start_epoch = 7


    # pdb.set_trace()
      
    # train the network
    if args.training_mode:

        gan.train(dataset=dataset,
                        num_workers=opt.num_works,
                        epochs=opt.sched.epochs,
                        batch_sizes=opt.sched.batch_sizes,
                        logger=logger,
                        output=output_dir,
                        loss_weight=opt.loss_weight,
                        num_samples=opt.num_samples,
                        feedback_factor=opt.feedback_factor,
                        checkpoint_factor=opt.checkpoint_factor,
                        start_epoch = start_epoch,
                        end_epoch = end_epoch)



    if args.visualizePWAVE:

        gan.visualizePWAVE(dataset=dataset, celeA_label=celeA_label, visual_size=16)



    # gen.test(dataset=dataset, num_examplar=args.num_examplar)




