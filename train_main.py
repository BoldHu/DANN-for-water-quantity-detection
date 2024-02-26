import random
import os
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from dataset.data_loader import GetLoader
from models.model import CNNModel
import numpy as np
from test import test
import matplotlib.pyplot as plt

def main():
    source_dataset_name = 'X_kla120.mat'
    source_dataset_labels_name = 'EQvec_kla120.mat'
    target_dataset_name = 'X_kla240.mat'
    target_dataset_labels_name = 'EQvec_kla240.mat'
    
    cudnn.benchmark = True
    lr = 1e-3
    batch_size = 32
    n_epoch = 3

    manual_seed = 42
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    # load data
    source_dataset = GetLoader(source_dataset_name, source_dataset_labels_name, transform=True)
    target_dataset = GetLoader(target_dataset_name, target_dataset_labels_name, transform=True)

    # create dataloaders
    dataloader_source = torch.utils.data.DataLoader(
        dataset=source_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)

    dataloader_target = torch.utils.data.DataLoader(
        dataset=target_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)
    
    print('read the data from the dataset')

    # load model
    my_net = CNNModel()

    # setup optimizer
    optimizer = optim.Adam(my_net.parameters(), lr=lr)
    # regression loss
    loss_reg = torch.nn.MSELoss()
    loss_domain = torch.nn.NLLLoss()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    if device.type == 'cuda':
        my_net = my_net.to(device)
        # loss_reg = loss_reg.to(device)
        # loss_domain = loss_domain.to(device)

    for p in my_net.parameters():
        p.requires_grad = True
        
    # training
    print('start training')
    len_dataloader = min(len(dataloader_source), len(dataloader_target))

    # Lists to store loss values for plotting
    err_s_label_list = []
    err_s_domain_list = []
    err_t_domain_list = []
    err_test_list = []
    r2_test_list = []
    rmse_test_list = []

    for epoch in range(n_epoch):
        i = 0
        dataloader_source_iter = iter(dataloader_source)
        dataloader_target_iter = iter(dataloader_target)
        # training mode
        my_net = my_net.train()
        while i < len_dataloader:
            
            # get data from source and target datasets
            data_source= next(dataloader_source_iter)
            source_feature, source_label = data_source
            data_target = next(dataloader_target_iter)
            target_feature, target_label = data_target

            p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            batch_size = len(source_feature)

            my_net.zero_grad()
            
            if device.type == 'cuda':
                print('Using device:', device)
                source_feature = source_feature.to(device)
                source_label = source_label.to(device)
                source_domain_label = torch.zeros(batch_size).to(device)
                
                target_domain_label = torch.ones(batch_size).to(device)  
                target_feature = target_feature.to(device)
                
                # change domain label to long
                source_domain_label = source_domain_label.long()
                target_domain_label = target_domain_label.long() 
                

            label_output, source_domain_output = my_net(input_data=source_feature, alpha=alpha)
            err_s_label = loss_reg(label_output, source_label)
            err_s_domain = loss_domain(source_domain_output, source_domain_label)

            _, target_domain_output = my_net(input_data=target_feature, alpha=alpha)
            err_t_domain = loss_domain(target_domain_output, target_domain_label)
            err = err_t_domain + err_s_domain + err_s_label
            err.backward()
            optimizer.step()

            print('epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' %
                (epoch, i, len_dataloader, err_s_label.item(),
                err_s_domain.item(), err_t_domain.item()))

            # Append loss values to lists for plotting
            err_s_label_list.append(err_s_label.item())
            err_s_domain_list.append(err_s_domain.item())
            err_t_domain_list.append(err_t_domain.item())

            i += 1
        # save model in saved_models directory
        torch.save(my_net, '{0}/DANN_model_epoch_{1}.pth'.format('saved_models', epoch))
        
        # test the model
        r2, rmse, regres_loss = test(my_net, loss_reg, epoch, device)
        err_test_list.append(regres_loss)
        r2_test_list.append(r2)
        rmse_test_list.append(rmse)

    print('done')

    # Plotting the loss values
    plt.figure(figsize=(10, 5))
    plt.plot(err_s_label_list, label='Source Label Loss')
    plt.plot(err_test_list, label='Target Lable Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.savefig('figures/train label losses.png')
    
    # plotting the domain loss
    plt.figure(figsize=(10, 5))
    plt.plot(err_s_domain_list, label='Source Domain Loss')
    plt.plot(err_t_domain_list, label='Target Domain Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Domain Losses')
    plt.legend()
    plt.savefig('figures/domain losses.png')
    
    # plotting the test results
    plt.figure(figsize=(10, 5))
    plt.plot(r2_test_list, label='R2 Score')
    plt.plot(rmse_test_list, label='RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Test Results')
    plt.legend()
    plt.savefig('figures/test results.png')

if __name__ == '__main__':
    main()
    print ('done')