import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils.dataset_npy import CSL_Continuous
from utils.data_path import path_data
from utils.dataset_slr import loader


def CSL(args):
    transform = transforms.Compose([transforms.Resize([args.rescale, args.rescale]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])
    train_set = CSL_Continuous(data_path=args.data_path, dict_path=args.dict_path,
                               corpus_path=args.corpus_path, train=True, transform=transform,
                               device=True)
    val_set = CSL_Continuous(data_path=args.data_path, dict_path=args.dict_path,
                             corpus_path=args.corpus_path, train=False, transform=transform,
                             device=True)
    test_set = CSL_Continuous(data_path=args.data_path, dict_path=args.dict_path,
                              corpus_path=args.corpus_path, train=False, transform=transform,
                              device=True)
    print("train-Dataset samples: {}".format(len(train_set)))
    print("val-Dataset samples: {}".format(len(val_set)))
    print("test-Dataset samples: {}".format(len(test_set)))
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=False)
    valid_dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=False)
    test_dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True,
                                 num_workers=args.num_workers, pin_memory=False)
    return train_dataloader, valid_dataloader, test_dataloader


def PHONEX2014(args):
    data = '../data/phoenix2014-release/phoenix-2014-multisigner'
    lookup = './data/slr_lookup.txt'
    train_path, valid_path, test_path = path_data(data_path=data, features_type=args.data_type,
                                                  hand_query=None)
    train_dataloader, train_size = loader(csv_file=train_path[1], root_dir=train_path[0],
                                          lookup=lookup, rescale=args.rescale,
                                          augmentation="store_true", batch_size=args.batch_size, num_workers=0,
                                          show_sample='store_true', istrain=True, fixed_padding=None,
                                          # hand_dir=train_path[2], hand_dir=None,
                                          data_stats="data_stats.pt", hand_stats=None, channels=3
                                          )

    # No data augmentation for valid data
    valid_dataloader, valid_size = loader(csv_file=valid_path[1], root_dir=valid_path[0],
                                          lookup=lookup,
                                          rescale=args.rescale, augmentation="store_true",
                                          batch_size=args.batch_size,
                                          num_workers=0, show_sample='store_true', istrain=False,
                                          fixed_padding=None, hand_dir=None,
                                          data_stats=None, hand_stats=None, channels=3
                                          )
    test_dataloader, test_size = loader(csv_file=test_path[1], root_dir=test_path[0],
                                        lookup=lookup,
                                        rescale=args.rescale, batch_size=args.batch_size,
                                        num_workers=0, istrain=False,
                                        data_stats=None, channels=3)
    print('Train dataset size: ' + str(train_size))
    print('Valid dataset size: ' + str(valid_size))
    print('Test dataset size: ' + str(test_size))
    return train_dataloader, valid_dataloader, test_dataloader
