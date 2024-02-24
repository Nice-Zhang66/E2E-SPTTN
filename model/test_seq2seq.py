from datetime import datetime

import torch
from sklearn.metrics import accuracy_score
from utils.tools import wer


def test_seq2seq(model, criterion, dataloader, device, epoch, logger, writer):
    model.eval()
    losses = []
    all_trg = []
    all_pred = []
    all_wer = []
    all_del = []
    all_sub = []
    all_ins = []

    with torch.no_grad():
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
            losses.append(loss.item())

            # collect labels & prediction
            prediction = torch.max(outputs, 1)[1]
            all_trg.extend(target)
            all_pred.extend(prediction)

            # compute wer
            # prediction: ((trg_len-1)*batch_size)
            # target: ((trg_len-1)*batch_size)
            batch_size = imgs.shape[0]
            prediction = prediction.view(-1, batch_size).permute(1, 0).tolist()
            target = target.view(-1, batch_size).permute(1, 0).tolist()
            wers = []
            Del_rate = []
            Sub_rate = []
            Ins_rate = []
            for i in range(batch_size):
                # add mask(remove padding, eos, sos)
                prediction[i] = [item for item in prediction[i] if item not in [0, 1, 2]]
                target[i] = [item for item in target[i] if item not in [0, 1, 2]]
                # wers.append(wer(target[i], prediction[i]))
                wers_list, werss = wer(target[i], prediction[i])
                # print(wers_list)
                wers.append(werss)
            all_wer.extend(wers)
            all_del.extend(Del_rate)
            all_sub.extend(Sub_rate)
            all_ins.extend(Ins_rate)

    # Compute the average loss & accuracy
    test_loss = sum(losses) / len(losses)
    all_trg = torch.stack(all_trg, dim=0)
    all_pred = torch.stack(all_pred, dim=0)
    test_acc = accuracy_score(all_trg.cpu().data.squeeze().numpy(), all_pred.cpu().data.squeeze().numpy())
    test_wer = sum(all_wer) / len(all_wer)
    # Log
    writer.add_scalars('Loss', {'test': test_loss}, epoch + 1)
    writer.add_scalars('WER', {'test': test_wer}, epoch + 1)
    writer.add_scalars('Accuracy', {'test': test_acc}, epoch + 1)
    logger.info("Average Test Loss: {:.6f} | Acc: {:.2f}%| WER: {:.4f}%".format(test_loss, test_acc * 100, test_wer))
    # logger.info("WER={:.4f}% DEL={:.4f}% INS={:.4f}% SUB={:.4f}%"
    #             .format(test_wer, test_wer_del, test_wer_ins, test_wer_sub))
    logger.info("*" * 60)


