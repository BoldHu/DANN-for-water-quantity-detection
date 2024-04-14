import random
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from dataset.data_loader import GetLoader
from models.model import CNNModel, CNNModel5, CNNModel7
import numpy as np
from reg_functions import reg_indicator
from plot_functions import *
from test_main import test_main
from remove_word import remove

def train(source_feature, source_label, target_feature, target_label):
    source_dataset_name = source_feature
    source_dataset_labels_name = source_label
    target_dataset_name = target_feature
    target_dataset_labels_name = target_label
    
    cudnn.benchmark = True
    lr = 1e-3
    batch_size = 32
    n_epoch = 100

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
        num_workers=32)

    dataloader_target = torch.utils.data.DataLoader(
        dataset=target_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=32)
    
    print('read the data from the dataset')

    # load model
    my_net = CNNModel7()

    # setup optimizer
    optimizer = optim.Adam(my_net.parameters(), lr=lr, weight_decay=1e-4)
    # regression loss
    loss_reg = torch.nn.MSELoss()
    loss_domain = torch.nn.NLLLoss()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type == 'cuda':
        my_net = my_net.to(device)
        loss_reg = loss_reg.to(device)
        loss_domain = loss_domain.to(device)

    for p in my_net.parameters():
        p.requires_grad = True
        
    # training
    print('start training')
    len_dataloader = min(len(dataloader_source), len(dataloader_target))

    # Lists to store loss values for plotting
    err_s_label_list = []
    err_s_domain_list = []
    err_t_domain_list = []
    err_t_label_list = []
    source_RMSE_list = []
    source_r2_list = []

    for epoch in range(n_epoch):
        i = 0
        dataloader_source_iter = iter(dataloader_source)
        dataloader_target_iter = iter(dataloader_target)
        # training mode
        my_net = my_net.train()
        # set the epoch loss values to zero
        err_s_label_epoch = 0
        err_t_domain_epoch = 0
        err_s_domain_epoch = 0
        source_RMSE = 0
        source_r2 = 0
        
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
            
            # calculate the RMSE and R2 for the source dataset
            reg_output, _ = my_net(input_data=source_feature, alpha=alpha)
            r2, rmse = reg_indicator(source_label, reg_output)

            source_RMSE += rmse.item()
            source_r2 += r2.item()
            err_s_domain_epoch += err_s_domain.item()
            err_t_domain_epoch += err_t_domain.item()
            err_s_label_epoch += err_s_label.item()

            i += 1
        _,_,err_t_label = test_main(source_feature=source_dataset_name, 
                                    source_label=source_dataset_labels_name, 
                                    target_feature=target_dataset_name, 
                                    target_label=target_dataset_labels_name,
                                    model=my_net)
        err_s_label_list.append(err_s_label_epoch / len_dataloader)
        err_s_domain_list.append(err_s_domain_epoch / len_dataloader)
        err_t_domain_list.append(err_t_domain_epoch / len_dataloader)
        err_t_label_list.append(err_t_label)
        source_RMSE_list.append(source_RMSE / len_dataloader)
        source_r2_list.append(source_r2 / len_dataloader)
        # print the results
        print('epoch: %d, source_RMSE: %f, source_r2: %f, err_s_label: %f, err_s_domain: %f, err_t_domain: %f, err_t_label: %f' %
                (epoch, source_RMSE / len_dataloader, source_r2 / len_dataloader, err_s_label_epoch / len_dataloader, err_s_domain_epoch / len_dataloader, err_t_domain_epoch / len_dataloader, err_t_label))
    
    source_dataset_name = remove(source_dataset_name)
    target_dataset_name = remove(target_dataset_name)
    # save model in saved_models directory
    torch.save(my_net, '{folder}/DANN7_model_{source_name}_{target_name}.pth'.format(folder='saved_models', source_name=source_dataset_name, target_name=target_dataset_name))
    # save the error values in a file with the source_name and target_name
    np.save('process_data/{source_name}_{target_name}_err_s_label_list.npy'.format(source_name=source_dataset_name, target_name=target_dataset_name), err_s_label_list)
    np.save('process_data/{source_name}_{target_name}_err_s_domain_list.npy'.format(source_name=source_dataset_name, target_name=target_dataset_name), err_s_domain_list)
    np.save('process_data/{source_name}_{target_name}_err_t_domain_list.npy'.format(source_name=source_dataset_name, target_name=target_dataset_name), err_t_domain_list)
    np.save('process_data/{source_name}_{target_name}_err_t_label_list.npy'.format(source_name=source_dataset_name, target_name=target_dataset_name), err_t_label_list)
    np.save('process_data/{source_name}_{target_name}_source_RMSE_list.npy'.format(source_name=source_dataset_name, target_name=target_dataset_name), source_RMSE_list)
    np.save('process_data/{source_name}_{target_name}_source_r2_list.npy'.format(source_name=source_dataset_name, target_name=target_dataset_name), source_r2_list)
    
    # print the figures
    plot_loss_label(err_s_label_list, 'Source Label Loss {0} {1}'.format(source_dataset_name, target_dataset_name))
    plot_loss_domain(err_s_domain_list, err_t_domain_list, 'Source Loss {0} {1}'.format(source_dataset_name, target_dataset_name))
    plot_loss_domain(err_s_label_list, err_t_label_list, 'Source and Target Label Loss {0} {1}'.format(source_dataset_name, target_dataset_name))
    

    print('done')
    return source_r2 / len_dataloader, source_RMSE / len_dataloader, err_s_label_epoch / len_dataloader
    