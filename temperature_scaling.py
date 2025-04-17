import torch
from torch import nn, optim
from torch.nn import functional as F
import argparse
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p

def TS_on_folder(gt_folder, pred_folder, TS):
    ## load pred, label
    
    
    ## TS
    nll_criterion = nn.CrossEntropyLoss().cuda()
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for input, label in valid_loader:
            input = input.cuda()
            logits = self.model(input)
            logits_list.append(logits)
            labels_list.append(label)
        logits = torch.cat(logits_list).cuda()
        labels = torch.cat(labels_list).cuda()
    optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

    def eval():
        optimizer.zero_grad()
        loss = nll_criterion(self.temperature_scale(logits), labels)
        loss.backward()
        return loss
    optimizer.step(eval)
    
    result_as_list = {}
    result_as_list['temperature'] = [temperature.detach().cpu().item()]  
    save_json(result_as_list, join(self.output_folder, f"temperature_{TS}.json"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_folder', type=str, help='folder with gt segmentations')
    parser.add_argument('pred_folder', type=str, help='folder with predicted segmentations')
    parser.add_argument('--TS', type=str, required=True, default='lbfgs', help='Temperature Scaling')


    args = parser.parse_args()
    TS_on_folder(args.gt_folder, args.pred_folder, args.TS)