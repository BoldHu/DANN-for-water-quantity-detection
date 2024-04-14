import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from reg_functions import reg_indicator
from dataset.data_loader import GetLoader
import random

def test_main(source_feature, source_label, target_feature, target_label, model):

    cuda = True
    cudnn.benchmark = True
    batch_size = 1344
    alpha = 0
    
    manual_seed = 42
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    
    # read the data from the dataset
    source_dataset_name = source_feature
    source_dataset_labels_name = source_label
    target_dataset_name = target_feature
    target_dataset_labels_name = target_label
        
    target_dataset = GetLoader(target_dataset_name, target_dataset_labels_name, transform=True)
    
    dataloader_test = torch.utils.data.DataLoader(
    dataset=target_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=32)
    len_dataloader = len(dataloader_test)
        
    # switch to evaluation mode
    my_net = model
    my_net.eval()
    
    # get the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == 'cuda':
        my_net = my_net.to(device)

    regres_loss_result = 0
    r2_result = 0
    rmse_result = 0
    loss_reg = torch.nn.MSELoss()
    
    with torch.no_grad():
        for idx_batch, (test_features, test_labels) in enumerate(dataloader_test):

            batch_size = len(test_labels)
            
            if device.type == 'cuda':
                test_features = test_features.to(device)
                test_labels = test_labels.to(device)

            reg_output, _ = my_net(input_data=test_features, alpha=alpha)
            regres_loss = loss_reg(reg_output, test_labels)
            r2, rmse = reg_indicator(test_labels, reg_output)
            
            # based on the batch_idx and batch_size update the results
            regres_loss_result += regres_loss.item()
            r2_result += r2.item()
            rmse_result += rmse.item()
            
    regres_loss_result /= len_dataloader
    r2_result /= len_dataloader
    rmse_result /= len_dataloader
    print('regres_loss: %f, r2: %f, rmse: %f' % (regres_loss_result, r2_result, rmse_result))
    
    return r2_result, rmse_result, regres_loss_result