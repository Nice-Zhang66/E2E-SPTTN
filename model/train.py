import time
from datetime import timedelta

import torch
from sklearn.metrics import accuracy_score
from utils.tools import wer


def train_seq2seq(model, criterion, optimizer, clip, dataloader, device, epoch, logger, log_interval, writer):
    model.train()
    losses = []
    all_trg = []
    all_pred = []
    all_wer = []
    start_time = time.time()
    # print(GPUtil.showUtilization())

    for batch_idx, data in enumerate(dataloader):
        target = torch.tensor(data[1])
        imgs = device.data_to_device(data[0])  # (batch x frames x channels x height x width )(4, T, 3, 128, 128)
        target = device.data_to_device(target)

        outputs = model(imgs, target)

        # target: (batch_size, trg len)
        # outputs: (trg_len, batch_size, output_dim)
        # skip sos
        output_dim = outputs.shape[-1]
        outputs = outputs[1:].contiguous().view(-1, output_dim)  # (8*bs, 1024)
        target = target.permute(1, 0)[1:].reshape(-1)  # 用于计算损失函数，不带起始的<sos>
        target = target.to(torch.int64)

        # compute the loss
        loss = criterion(outputs, target)
        losses.append(float(loss.item()))

        # compute the accuracy
        prediction = torch.max(outputs, 1)[1]
        y_pred = prediction.cpu().data.squeeze().numpy()
        y_true = target.cpu().data.squeeze().numpy()
        # score = accuracy_score(y_true, y_pred)
        # score = score*100
        all_trg.extend(target)
        all_pred.extend(prediction)

        # compute wer
        # prediction: ((trg_len-1)*batch_size)
        # target: ((trg_len-1)*batch_size)
        batch_size = imgs.shape[0]
        prediction = prediction.view(-1, batch_size).permute(1, 0).tolist()
        target = target.view(-1, batch_size).permute(1, 0).tolist()
        wers = []
        for i in range(batch_size):
            # add mask(remove padding, sos, eos)
            prediction[i] = [item for item in prediction[i] if item not in [0, 1, 2]]
            target[i] = [item for item in target[i] if item not in [0, 1, 2]]
            wers_list, werss = wer(target[i], prediction[i])
            # print(wers_list)
            wers.append(werss)
        all_wer.extend(wers)

        # backward & optimize
        optimizer.zero_grad()  # 先将梯度归零
        loss.backward()  # 反向传播计算得到每个参数的梯度值
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()  # 通过梯度下降执行一步网络参数更新

        # 添加的第一条日志：损失函数-全局迭代次数
        # writer.add_scalar("train loss", loss.item(), global_step=batch_idx + 1)
        # 添加第二条日志：正确率-全局迭代次数
        # writer.add_scalar("test accuary", score.item(), global_step=batch_idx + 1)
        # 添加第二条日志：正确率-全局迭代次数
        # writer.add_scalar("test wer", werss, global_step=batch_idx + 1)

        # images = imgs.squeeze(2)
        # writer.add_images("train image sample", images, global_step=batch_idx+1)
        # writer.add_graph(model, input_to_model=None, verbose=False, **kwargs)
        for name, param in model.named_parameters():
            writer.add_histogram(name, param.data.cpu().numpy(), global_step=batch_idx + 1)

        total_time = time.time() - start_time
        if (batch_idx + 1) % log_interval == 0:
            logger.info("epoch {:3d} | iteration {:5d} | Loss {:.6f} | WER {:.4f}% | Time : {:.2f}s, "
                        "{:.2f}ms/segment".format(epoch + 1, batch_idx + 1, loss.item(), sum(wers) / len(wers),
                                                  total_time, 1000 * total_time / (batch_idx + 1)))
    total_time = time.time() - start_time
    total_time_str = str(timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    # Compute the average loss & accuracy
    training_loss = sum(losses)/len(losses)
    all_trg = torch.stack(all_trg, dim=0)
    all_pred = torch.stack(all_pred, dim=0)
    training_acc = accuracy_score(all_trg.cpu().data.squeeze().numpy(), all_pred.cpu().data.squeeze().numpy())
    training_wer = sum(all_wer)/len(all_wer)
    # Log
    # writer.add_scalars('Loss', {'train': training_loss}, epoch+1)
    # writer.add_scalars('Accuracy', {'train': training_acc}, epoch+1)
    # writer.add_scalars('WER', {'train': training_wer}, epoch+1)
    logger.info("Average Training Loss of Epoch {}: {:.6f} | Acc: {:.2f}% | WER {:.2f}%".format(epoch+1, training_loss, training_acc*100, training_wer))
