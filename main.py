import os
import torch
import torch.nn as nn
import numpy as np
import hiddenlayer as h
import torch.optim as optim
import logging

from torch.utils.data import DataLoader

import utils
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from utils.dataloader_video import BaseFeeder

from model.test_seq2seq import test_seq2seq
from model.train import train_seq2seq

from model.validation import val_seq2seq
from model.Transformer import Transformer
import matplotlib.pyplot as plt
from utils.dataset import CSL

plt.ion()


class Recognition:
    def __init__(self, args):
        self.data_loader = {}
        self.dataset = {}
        self.gloss_dict = np.load('./data/phoenix2014/gloss_dict.npy', allow_pickle=True).item()
        self.feeder = BaseFeeder(self.gloss_dict)
        self.arg = args
        self.device = utils.GpuDataParallel()
        log_path = "model/log/seq2seq_{:%Y-%m-%d_%H-%M-%S}.log".format(datetime.now())
        sum_path = "./data/log/slr_seq2seq2_{:%Y-%m-%d_%H-%M-%S}".format(datetime.now())
        logging.basicConfig(level=logging.INFO, format='%(message)s',
                            handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
        self.logger = logging.getLogger('E2E-STTN SLR')
        self.writer = SummaryWriter(sum_path)
        self.criterion = nn.CrossEntropyLoss()

    def start(self):
        # 启用GPU
        self.device.set_device(self.arg.device)
        # Set the random seed manually for reproducibility.
        torch.manual_seed(args.seed)
        self.logger.info('Logging to file...')
        # Log to file & tensorboard writer
        if args.datatype == 'CSL':
            # CSL数据集
            train_dataloader, valid_dataloader, test_dataloader = CSL(args)
        else:
            self.load_data()
            # Pass the annotation + image sequences locations
            # train_dataloader, valid_dataloader, test_dataloader = PHONEX2014(args)
            train_dataloader = self.data_loader['train']
            valid_dataloader = self.data_loader['dev']
            test_dataloader = self.data_loader["test"]

        model = Transformer(args.rescale, 5000)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        model = self.model_to_device(model)
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
        print('Trainable Parameters: %.3fM' % parameters)

        # Start training
        self.logger.info("Training Started".center(60, '#'))
        print("Started Time:", format(datetime.now()))
        seq_model_list = []
        for epoch in range(args.epochs):
            # Train the model
            train_seq2seq(model, self.criterion, optimizer, args.clip,  train_dataloader, self.device, epoch,
                          self.logger, args.log_interval, self.writer)

            # Validate the model
            val_seq2seq(model, self.criterion, valid_dataloader, self.device, epoch, self.logger, self.writer)

            # Test the model
            test_seq2seq(model, self.criterion, test_dataloader, self.device, epoch, self.logger, self.writer)
            # Save model
            save_model = epoch % self.arg.save_interval == 0
            if save_model:
                save_path = os.path.join(args.model_path, "slr_seq2seq_epoch{:03d}.pth".format(epoch + 1))
                torch.save(model.state_dict(), save_path)
                seq_model_list.append(save_path)
                print("seq_model_list", seq_model_list)
                self.logger.info("Epoch {} Model Saved".format(epoch + 1).center(60, '#'))
        vis_graph = h.build_graph(model, (torch.rand([4, 48, 3, 224, 224]), torch.rand([4, 9])))  # 获取绘制图像的对象
        vis_graph.theme = h.graph.THEMES["blue"].copy()  # 指定主题颜色
        vis_graph.save("./demo1.png")  # 保存图像的路径

    def model_to_device(self, model):
        model = model.to(self.device.output_device)
        if len(self.device.gpu_list) > 1:
            model = nn.DataParallel(
                model,
                device_ids=self.device.gpu_list,
                output_device=self.device.output_device)
        model.cuda()
        return model

    def load_data(self):
        print("Loading data")
        dataset_list = zip(["train", "train_eval", "dev", "test"], [True, False, False, False])
        for idx, (mode, train_flag) in enumerate(dataset_list):
            # arg = self.arg.feeder_args
            # arg["prefix"] = self.arg.dataset_info['dataset_root']
            # arg["mode"] = mode.split("_")[0]
            # arg["transform_mode"] = train_flag
            self.dataset[mode] = self.feeder
            self.data_loader[mode] = self.build_dataloader(self.dataset[mode], mode, train_flag)
        print("Loading data finished.")

    def build_dataloader(self, dataset, mode, train_flag):
        return DataLoader(
            dataset,
            batch_size=self.arg.batch_size if mode == "train" else self.arg.test_batch_size,
            shuffle=train_flag,
            drop_last=train_flag,
            num_workers=self.arg.num_workers,  # if train_flag else 0
            collate_fn=self.feeder.collate_fn,
        )


if __name__ == '__main__':
    sparser = utils.get_parser()
    args = sparser.parse_args()
    recognition = Recognition(args)
    recognition.start()
