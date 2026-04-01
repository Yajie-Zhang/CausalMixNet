import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import logging
import numpy as np


# validate the algorithm by AUC, accuracy and f1 score on val/test datasets

def algorithm_validate(algorithm, data_loader, epoch, val_type, device,best_model=None,writer=None):
    if best_model is not None:
        path = torch.load(best_model, map_location=device)
        algorithm.load_state_dict(path, strict=True)

    algorithm.eval()
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        softmax = torch.nn.Softmax(dim=1)
        loss = 0
        label_list = []
        output_list = []
        pred_list = []

        for _,image,label in data_loader:
            image = image.to(device)
            label = label.to(device).long()

            _,_,output = algorithm(image)
            # loss += criterion(output, label).item()

            _, pred = torch.max(output, 1)
            output_sf = softmax(output)

            label_list.append(label.cpu().data.numpy())
            pred_list.append(pred.cpu().data.numpy())
            output_list.append(output_sf.cpu().data.numpy())

        label = [item for sublist in label_list for item in sublist]
        pred = [item for sublist in pred_list for item in sublist]
        output = [item for sublist in output_list for item in sublist]

        # print(label.shape)
        # print(label)
        # print(pred)

        acc = accuracy_score(label, pred)
        f1 = f1_score(label, pred, average='macro')

        auc_ovo = roc_auc_score(label, output, average='macro', multi_class='ovo')

        loss = loss / len(data_loader)

        if val_type in ['val', 'test']:
            writer.add_scalar('info/{}_acc'.format(val_type), acc, epoch)
            writer.add_scalar('info/{}_auc'.format(val_type), auc_ovo, epoch)
            writer.add_scalar('info/{}_f1'.format(val_type), f1, epoch)
            logging.info('{} - epoch: {}, loss: {}, acc: {}, auc: {}, F1: {}.'.format
                         (val_type, epoch, loss, acc, auc_ovo, f1))

    algorithm.train()
    return acc,f1,auc_ovo


def BRACS_label_transfer(label):
    label[label < 3] = 0
    label[label > 0] = 1
    return label


def BreakHis_label_transfer(label):
    label[label < 4] = 0
    label[label > 0] = 1
    return label


def algorithm_validate_he(algorithm, data_loader, epoch, val_type, device,best_model=None,writer=None):
    if best_model is not None:
        path = torch.load(best_model, map_location=device)
        algorithm.load_state_dict(path, strict=True)
    algorithm.eval()
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        softmax = torch.nn.Softmax(dim=1)
        loss = 0
        label_list = []
        output_list = []
        pred_list = []

        for _,image,label in data_loader:
            image = image.to(device)
            label = label.to(device).long()

            _,_,output = algorithm(image)
            # loss += criterion(output, label).item()

            _, pred = torch.max(output, 1)
            output_sf = softmax(output)

            label_list.append(label.cpu().data.numpy())
            pred_list.append(pred.cpu().data.numpy())
            output_list.append(output_sf.cpu().data.numpy())

        label = [item for sublist in label_list for item in sublist]
        pred = [item for sublist in pred_list for item in sublist]
        output = [item for sublist in output_list for item in sublist]

        if val_type == 'val':
            multi_acc = accuracy_score(label, pred)
            multi_f1 = f1_score(label, pred, average='macro')
            label = torch.tensor(np.array(label))
            pred = torch.tensor(np.array(pred))
            label = BRACS_label_transfer(label)
            pred = BRACS_label_transfer(pred)
            label = list(label.numpy())
            pred = list(pred.numpy())
            by_acc = accuracy_score(label, pred)
            by_pr = precision_score(label, pred)
            by_rc = recall_score(label, pred)
            by_f1 = f1_score(label, pred)
            if val_type in ['val', 'test']:
                writer.add_scalar('info/{}_multi_acc'.format(val_type), multi_acc, epoch)
                writer.add_scalar('info/{}_multi_f1'.format(val_type), multi_f1, epoch)
                writer.add_scalar('info/{}_by_acc'.format(val_type), by_acc, epoch)
                writer.add_scalar('info/{}_by_pr'.format(val_type), by_pr, epoch)
                writer.add_scalar('info/{}_by_rc'.format(val_type), by_rc, epoch)
                writer.add_scalar('info/{}_by_f1'.format(val_type), by_f1, epoch)
                logging.info(
                    '{} - epoch: {}, multi_acc: {}, multi_f1: {}, by_acc: {}, by_pr:{},by_rc:{}, F1: {}.'.format
                    (val_type, epoch, multi_acc, multi_f1, by_acc, by_pr, by_rc, by_f1))
            algorithm.train()
            return multi_acc,multi_f1,by_acc,by_pr,by_rc,by_f1
        else:
            multi_acc = accuracy_score(label, pred)
            multi_f1 = f1_score(label, pred, average='macro')
            label = torch.tensor(np.array(label))
            pred = torch.tensor(np.array(pred))
            label = BRACS_label_transfer(label)
            pred = BRACS_label_transfer(pred)
            label = list(label.numpy())
            pred = list(pred.numpy())
            by_acc = accuracy_score(label, pred)
            by_pr = precision_score(label, pred)
            by_rc = recall_score(label, pred)
            by_f1 = f1_score(label, pred)
            if val_type in ['val', 'test']:
                writer.add_scalar('info/{}_multi_acc'.format(val_type), multi_acc, epoch)
                writer.add_scalar('info/{}_multi_f1'.format(val_type), multi_f1, epoch)
                writer.add_scalar('info/{}_by_acc'.format(val_type), by_acc, epoch)
                writer.add_scalar('info/{}_by_pr'.format(val_type), by_pr, epoch)
                writer.add_scalar('info/{}_by_rc'.format(val_type), by_rc, epoch)
                writer.add_scalar('info/{}_by_f1'.format(val_type), by_f1, epoch)
                logging.info(
                    '{} - epoch: {}, multi_acc: {}, multi_f1: {}, by_acc: {}, by_pr:{},by_rc:{}, F1: {}.'.format
                    (val_type, epoch, multi_acc, multi_f1, by_acc, by_pr, by_rc, by_f1))
            algorithm.train()
            return multi_acc, multi_f1, by_acc, by_pr, by_rc, by_f1


def algorithm_validate_he_break(algorithm, data_loader, epoch, val_type, device,best_model=None,writer=None):
    if best_model is not None:
        path = torch.load(best_model, map_location=device)
        algorithm.load_state_dict(path, strict=True)
    algorithm.eval()
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        softmax = torch.nn.Softmax(dim=1)
        loss = 0
        label_list = []
        output_list = []
        pred_list = []

        for _,image,label in data_loader:
            image = image.to(device)
            label = label.to(device).long()

            _,_,output = algorithm(image)
            # loss += criterion(output, label).item()

            _, pred = torch.max(output, 1)
            output_sf = softmax(output)

            label_list.append(label.cpu().data.numpy())
            pred_list.append(pred.cpu().data.numpy())
            output_list.append(output_sf.cpu().data.numpy())

        label = [item for sublist in label_list for item in sublist]
        pred = [item for sublist in pred_list for item in sublist]
        output = [item for sublist in output_list for item in sublist]

        if val_type == 'val':
            multi_acc = accuracy_score(label, pred)
            multi_f1 = f1_score(label, pred, average='macro')
            label = torch.tensor(np.array(label))
            pred = torch.tensor(np.array(pred))
            label = BreakHis_label_transfer(label)
            pred = BreakHis_label_transfer(pred)
            label = list(label.numpy())
            pred = list(pred.numpy())
            by_acc = accuracy_score(label, pred)
            by_pr = precision_score(label, pred)
            by_rc = recall_score(label, pred)
            by_f1 = f1_score(label, pred)
            if val_type in ['val', 'test']:
                writer.add_scalar('info/{}_multi_acc'.format(val_type), multi_acc, epoch)
                writer.add_scalar('info/{}_multi_f1'.format(val_type), multi_f1, epoch)
                writer.add_scalar('info/{}_by_acc'.format(val_type), by_acc, epoch)
                writer.add_scalar('info/{}_by_pr'.format(val_type), by_pr, epoch)
                writer.add_scalar('info/{}_by_rc'.format(val_type), by_rc, epoch)
                writer.add_scalar('info/{}_by_f1'.format(val_type), by_f1, epoch)
                logging.info(
                    '{} - epoch: {}, multi_acc: {}, multi_f1: {}, by_acc: {}, by_pr:{},by_rc:{}, F1: {}.'.format
                    (val_type, epoch, multi_acc, multi_f1, by_acc, by_pr, by_rc, by_f1))
            algorithm.train()
            return multi_acc,multi_f1,by_acc,by_pr,by_rc,by_f1
        else:
            label = torch.tensor(np.array(label))
            pred = torch.tensor(np.array(pred))
            label = BRACS_label_transfer(label)
            pred = BreakHis_label_transfer(pred)
            select_index=torch.where(label<2)[0]
            label=label[select_index]
            pred=pred[select_index]
            label = list(label.numpy())
            pred = list(pred.numpy())
            by_acc = accuracy_score(label, pred)
            by_pr = precision_score(label, pred)
            by_rc = recall_score(label, pred)
            by_f1 = f1_score(label, pred)
            if val_type in ['val', 'test']:
                # writer.add_scalar('info/{}_multi_acc'.format(val_type), multi_acc, epoch)
                # writer.add_scalar('info/{}_multi_f1'.format(val_type), multi_f1, epoch)
                writer.add_scalar('info/{}_by_acc'.format(val_type), by_acc, epoch)
                writer.add_scalar('info/{}_by_pr'.format(val_type), by_pr, epoch)
                writer.add_scalar('info/{}_by_rc'.format(val_type), by_rc, epoch)
                writer.add_scalar('info/{}_by_f1'.format(val_type), by_f1, epoch)
                logging.info(
                    '{} - epoch: {},by_acc: {}, by_pr:{},by_rc:{}, F1: {}.'.format
                    (val_type, epoch, by_acc, by_pr, by_rc, by_f1))
            algorithm.train()
            return by_acc,by_pr,by_rc,by_f1


def mean_average_precision(algorithm,query_loader,base_loader,args,best_model=None,writer=None):
    device=torch.device(args.device)
    if best_model is not None:
        path = torch.load(best_model, map_location=device)
        algorithm.load_state_dict(path, strict=True)
    d_ft=512
    algorithm.eval()
    with torch.no_grad():
        query_code=torch.zeros(len(query_loader.dataset),d_ft).to(device)
        query_label=torch.zeros(len(query_loader.dataset),200).to(device)

        for ind,image,label in query_loader:
            image = image.to(device)
            label = label.to(device).long()
            label=torch.tensor(np.eye(200, dtype=np.uint8)[label.cpu().numpy()]).float().to(device)

            _,ft,output = algorithm(image)
            query_code[ind]=ft.data
            query_label[ind]=label
        retrieval_code = torch.zeros(len(base_loader.dataset), d_ft).to(device)
        retrieval_label= torch.zeros(len(base_loader.dataset), 200).to(device)

        for ind, image, label in base_loader:
            image = image.to(device)
            label = label.to(device).long()
            label = torch.tensor(np.eye(200, dtype=np.uint8)[label.cpu().numpy()]).float().to(device)

            _, ft, output = algorithm(image)
            retrieval_code[ind] = ft.data
            retrieval_label[ind] = label

    retrieval_code=torch.nn.functional.normalize(retrieval_code,dim=1)
    query_code=torch.nn.functional.normalize(query_code,dim=1)
    query_code = query_code.cpu().detach().numpy()
    query_label = query_label.cpu().detach().numpy()
    retrieval_code = retrieval_code.cpu().detach().numpy()
    retrieval_label = retrieval_label.cpu().detach().numpy()

    query_num=query_code.shape[0]
    # retrieval_code=np.sign(retrieval_code)
    # query_code=np.sign(query_code)

    sim=np.dot(retrieval_code,query_code.T)   
    ids=np.argsort(-sim,axis=0)

    all_res=[]
    for R in [5,10,20,30,40,50]:
        APx = []

        for i in range(query_num):
            label = query_label[i, :]
            label[label == 0] = -1  
            idx = ids[:, i]
            imatch = np.sum(retrieval_label[idx[0:R], :] == label, axis=1) > 0  # retrieval_data是否和这个query有同样的标签。
            relevant_num = np.sum(imatch)
            Lx = np.cumsum(imatch)
            Px = Lx.astype(float) / np.arange(1, R + 1, 1)
            if relevant_num != 0:
                APx.append(np.sum(Px * imatch) / relevant_num)
            else:
                APx.append(0.0)
        all_res.append(np.mean(np.array(APx)))
    # writer.add_scalar('info/{}_mAP'.format('val'),torch.tensor(np.array(all_res)))
    logging.info(all_res)


    return all_res