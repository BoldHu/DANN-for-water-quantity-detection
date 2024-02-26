import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from reg_functions import reg_indicator
from dataset.data_loader import GetLoader

def test(my_net, loss_reg, epoch, device):

    cuda = True
    cudnn.benchmark = True
    batch_size = 32
    alpha = 0

    my_net = my_net.eval()
    
    if device == 'cuda':
        my_net = my_net.to(device)
    
    test_dataset_name = 'X_kla240.mat'
    test_dataset_labels_name = 'EQvec_kla240.mat'
    test_dataset = GetLoader(test_dataset_name, test_dataset_labels_name, transform=True)
    
    dataloader_test = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)
    
    len_dataloader = len(dataloader_test)
    
    regres_loss_result = 0
    r2_result = 0
    rmse_result = 0
    
    with torch.no_grad():
        for idx_batch, (test_features, test_labels) in enumerate(dataloader_test):

            batch_size = len(test_labels)
            
            if device == 'cuda':
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
    print('epoch: %d, regres_loss: %f, r2: %f, rmse: %f' % (epoch, regres_loss_result, r2_result, rmse_result))
    
    return r2_result, rmse_result, regres_loss_result
