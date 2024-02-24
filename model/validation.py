import torch
from sklearn.metrics import accuracy_score
from utils.tools import wer


def val_seq2seq(model, criterion, dataloader, device, epoch, logger, writer):
    model.eval()
    losses = []
    all_trg = []
    all_pred = []
    all_wer = []

    for batch_idx, data in enumerate(dataloader):

        target = torch.tensor(data[1])
        imgs = device.data_to_device(data[0])  # (batch x frames x channels x height x width )(4, T, 3, 128, 128)
        target = device.data_to_device(target)

        with torch.no_grad():
            outputs = model(image, target).to(device)

        # target: (batch_size, trg len)
        # outputs: (trg_len, batch_size, output_dim)
        # skip sos
        output_dim = outputs.shape[-1]
        outputs = outputs[1:].contiguous().view(-1, output_dim)
        target = target.permute(1, 0)[1:].reshape(-1)

        # compute the loss
        loss = criterion(outputs, target.long())
        losses.append(loss.item())

        # compute the accuracy
        prediction = torch.max(outputs, 1)[1]
        score = accuracy_score(target.cpu().data.squeeze().numpy(), prediction.cpu().data.squeeze().numpy())
        all_trg.extend(target)
        all_pred.extend(prediction)

        # compute wer
        # prediction: ((trg_len-1)*batch_size)
        # target: ((trg_len-1)*batch_size)
        batch_size = image.shape[0]
        prediction = prediction.view(-1, batch_size).permute(1,0).tolist()
        target = target.view(-1, batch_size).permute(1,0).tolist()
        wers = []
        for i in range(batch_size):
            # add mask(remove padding, eos, sos)
            prediction[i] = [item for item in prediction[i] if item not in [0,1,2]]
            target[i] = [item for item in target[i] if item not in [0,1,2]]
            # wers.append(wer(target[i], prediction[i]))
            wers_list, werss = wer(target[i], prediction[i])
            # print(wers_list)
            wers.append(werss)
        all_wer.extend(wers)

    # Compute the average loss & accuracy
    validation_loss = sum(losses)/len(losses)
    all_trg = torch.stack(all_trg, dim=0)
    all_pred = torch.stack(all_pred, dim=0)
    validation_acc = accuracy_score(all_trg.cpu().data.squeeze().numpy(), all_pred.cpu().data.squeeze().numpy())
    validation_wer = sum(all_wer)/len(all_wer)
    # Log
    writer.add_scalars('Loss', {'validation': validation_loss}, epoch+1)
    writer.add_scalars('Accuracy', {'validation': validation_acc}, epoch+1)
    writer.add_scalars('WER', {'validation': validation_wer}, epoch+1)
    logger.info("Average Validation Loss of Epoch {}: {:.6f} | Acc: {:.2f}% | WER: {:.2f}%".format(epoch+1, validation_loss, validation_acc*100, validation_wer))
