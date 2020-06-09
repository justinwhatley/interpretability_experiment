import torch
import torchvision
from tqdm import tqdm

# Model / Data Set Choice
from project.experiments.datasets.linear_reg_dataset import get_test_loader
from project.experiments.models.lr_model import get_model



def main():
    # ============================================================================================
    # Setup
    # ============================================================================================
    




#     # Training Settings
#     model_path = f'../training_scripts/saved_models/{file_name}_fold_0_best_acc_state.pt'

#     # ------------------------------------------------------------------------------------------------------
#     # GET MODEL
#     # ------------------------------------------------------------------------------------------------------
#     print('LOADING MODEL FROM: [{}]'.format(model_path))

#     # Get Model
    
    
#     # ------------------------------------------------------------------------------------------------------
#     # TEST SET PREDICTIONS
#     # ------------------------------------------------------------------------------------------------------
#     # Get test dataloader
#     test_dataloader = get_test_loader()

#     # Prediction
#     acc = make_prediction(net, device, test_dataloader, fp_16)
#     print('Acc: ', acc)


# def make_prediction(net, device, dataloader, fp16):
#     all_y_true = []
#     all_y_pred = []

#     net.eval()
#     with torch.no_grad():
#         for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
#             # Data
# #             x, y = data
# #             x, y = x.to(device), y.to(device)
# #             if fp16:
# #                 x = x.half()

# #             # Prediction
# #             y_pred = net(x).float()
# #             _, predicted = torch.max(y_pred, 1)

# #             all_y_true.extend(y.cpu().detach().numpy())
# #             all_y_pred.extend(predicted.cpu().detach().numpy())

#     assert(len(all_y_pred) == len(all_y_true))
#     acc = sum(np.array(all_y_true) == np.array(all_y_pred)) / len(all_y_pred)
#     return acc


if __name__ == '__main__':
    main()