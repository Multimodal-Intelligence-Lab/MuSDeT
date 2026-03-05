import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.modules.hypercomplex.hypercomplex_layers import PHMLinear

class eyeBase(nn.Module): 
    "Base for the eye Modality."
    def __init__(self, units=128):
        super(eyeBase, self).__init__()  # call the parent constructor
        self.flat = nn.Flatten()
        self.D1 = nn.Linear(600*4, units)
        self.BN1 = nn.BatchNorm1d(units)
        self.D2 = nn.Linear(units, units)
        self.BN2 = nn.BatchNorm1d(units)
        self.D3 = nn.Linear(units, units)

    def forward(self, inputs):
        x = self.flat(inputs)
        x = self.D1(x)
        x = F.relu(self.BN1(x))
        x = self.D2(x)
        x = F.relu(self.BN2(x))
        x = F.relu(self.D3(x))
        return x

class GSRBase(nn.Module):  
    "Base for the GSR Modality."
    def __init__(self, units=128):
        super(GSRBase, self).__init__()  # call the parent constructor
        self.D1 = nn.Linear(1280, units)
        self.BN1 = nn.BatchNorm1d(units)
        self.D2 = nn.Linear(units, units)

    def forward(self, inputs):
        x = self.D1(inputs).squeeze(1)
        x = F.relu(self.BN1(x))
        x = F.relu(self.D2(x))
        return x

class EEGBase(nn.Module):  
    "Base for the EEG Modality."
    def __init__(self, units=1024):
        super(EEGBase, self).__init__()  # call the parent constructor
        self.flat = nn.Flatten()
        self.D1 = nn.Linear(1280*10, units)
        self.BN1 = nn.BatchNorm1d(units)
        self.D2 = nn.Linear(units, units)
        self.BN2 = nn.BatchNorm1d(units)
        self.D3 = nn.Linear(units, units)

    def forward(self, inputs):
        x = self.flat(inputs)
        x = self.D1(x)
        x = F.relu(self.BN1(x))
        x = self.D2(x)
        x = F.relu(self.BN2(x))
        x = F.relu(self.D3(x))
        return x

class ECGBase(nn.Module): 
    "Base for the ECG Modality."
    def __init__(self, in_features, units=512):
        super(ECGBase, self).__init__()  # call the parent constructor
        self.flat = nn.Flatten()
        # self.D1 = nn.Linear(1280*3, units)
        self.D1 = nn.Linear(in_features, units)

        self.BN1 = nn.BatchNorm1d(units)
        self.D2 = nn.Linear(units, units)
        self.BN2 = nn.BatchNorm1d(units)
        self.D3 = nn.Linear(units, units)

    def forward(self, inputs):
        x = self.flat(inputs)
        x = self.D1(x)
        x = F.relu(self.BN1(x))
        x = self.D2(x)
        x = F.relu(self.BN2(x))
        x = F.relu(self.D3(x))
        return x

# class HyperFuseNet(nn.Module): 
#     """Head class that learns from all bases.
#     First dense layer has the name number of units as all bases
#     combined have as outputs."""
#     def __init__(self, dropout_rate, units=1024, n=4):
#         super(HyperFuseNet, self).__init__()  # call the parent constructor
#         self.eye = eyeBase()
#         self.gsr = GSRBase()
#         self.eeg = EEGBase()
#         self.ecg = ECGBase()
#         self.drop = nn.Dropout(dropout_rate)
#         self.D1 = PHMLinear(n, 1792, 1792)
#         self.BN1 = nn.BatchNorm1d(1792)
#         self.D2 = PHMLinear(n, 1792, units)
#         self.BN2 = nn.BatchNorm1d(units)
#         self.D3 = PHMLinear(n, units, units//2)
#         self.BN3 = nn.BatchNorm1d(units//2)
#         self.D4 = PHMLinear(n, units//2, units//4)
#         self.out_3 = nn.Linear(units//4, 3)
    
#     def get_features(self, eye, gsr, eeg, ecg, level='encoder'):
#         assert level in ['encoder', 'classifier']
#         eye_out = self.eye(eye)
#         gsr_out = self.gsr(gsr)
#         eeg_out = self.eeg(eeg)
#         ecg_out = self.ecg(ecg)
#         concat = torch.cat([eye_out, gsr_out, eeg_out, ecg_out], dim=1)
#         if level == 'encoder':
#             return concat
#         x = F.relu(self.BN1(self.D1(concat)))
#         x = F.relu(self.BN2(self.D2(x)))
#         x = F.relu(self.BN3(self.D3(x)))
#         x = F.relu(self.D4(x))
#         return x

#     def forward(self, eye, gsr, eeg, ecg):
#         eye_out = self.eye(eye)
#         gsr_out = self.gsr(gsr)
#         eeg_out = self.eeg(eeg)
#         ecg_out = self.ecg(ecg)
#         concat = torch.cat([eye_out, gsr_out, eeg_out, ecg_out], dim=1)
#         x = self.D1(concat)
#         x = F.relu(self.BN1(x))
#         x = self.D2(x)
#         x = F.relu(self.BN2(x))
#         x = self.drop(x)
#         x = self.D3(x)
#         x = F.relu(self.BN3(x))
#         x = F.relu(self.D4(x))
#         out = self.out_3(x)  # Softmax would be applied directly by CrossEntropyLoss, because labels=classes
#         return out
    
class HyperFuseNet(nn.Module): 
    """Head class that learns from all bases.
    First dense layer has the name number of units as all bases
    combined have as outputs."""
    # def __init__(self, dropout_rate, units=1024, n=4):
    def __init__(self, model_cfg, seq_dims, data_dims, n_classes, units=1024, n=4, n_eye=4, n_gsr=1, n_eeg=10, n_ecg=3):
        super(HyperFuseNet, self).__init__()  # call the parent constructor
        # self.eye = eyeBase()
        # self.gsr = GSRBase()
        # self.eeg = EEGBase()
        # self.ecg = ECGBase()
        self.data_dims = data_dims
        self.seq_dims = seq_dims
        self.modality_keys = list(self.data_dims.keys())
        PHBase_dict = {}

        for k in self.modality_keys:
            if 'eeg' in k.lower():
                PHBase_dict[f'{k}'] = EEGBase
            elif 'ecg' in k.lower():
                PHBase_dict[f'{k}'] = ECGBase
            # elif 'gsr' in k.lower():
            #     PHBase_dict[f'{k}'] = GSRPHBase
            else:
                PHBase_dict[f'{k}'] = ECGBase


        self.PHbase = nn.ModuleDict({
            # m: PHBase_dict[m](n=self.seq_dims[m], in_features=self.data_dims[m])
            m: PHBase_dict[m](in_features=self.data_dims[m]*self.seq_dims[m])
            for m in self.modality_keys
        })

        dropout_rate= 0.5

        self.drop = nn.Dropout(dropout_rate)
        # self.D1 = PHMLinear(n, 1792, 1792)
        # self.D1 = PHMLinear(n, 3072, 1792) # WESAD
        concat_dim = len(self.modality_keys) * 512  # each ECGBase outputs 512 units
        self.D1 = PHMLinear(n, concat_dim, 1792)
        self.BN1 = nn.BatchNorm1d(1792)
        self.D2 = PHMLinear(n, 1792, units)
        self.BN2 = nn.BatchNorm1d(units)
        self.D3 = PHMLinear(n, units, units//2)
        self.BN3 = nn.BatchNorm1d(units//2)
        self.D4 = PHMLinear(n, units//2, units//4)
        self.out_3 = nn.Linear(units//4, n_classes)
    
    def get_features(self, eye, gsr, eeg, ecg, level='encoder'):
        assert level in ['encoder', 'classifier']
        eye_out = self.eye(eye)
        gsr_out = self.gsr(gsr)
        eeg_out = self.eeg(eeg)
        ecg_out = self.ecg(ecg)
        concat = torch.cat([eye_out, gsr_out, eeg_out, ecg_out], dim=1)
        if level == 'encoder':
            return concat
        x = F.relu(self.BN1(self.D1(concat)))
        x = F.relu(self.BN2(self.D2(x)))
        x = F.relu(self.BN3(self.D3(x)))
        x = F.relu(self.D4(x))
        return x

    # def forward(self, eye, gsr, eeg, ecg):
    def forward(self, x_dict):
        X = {}

        for m in self.modality_keys:
            x = x_dict[m].flatten(start_dim=1)#.transpose(1, 2)          # (B, C, D)
            x = self.PHbase[m](x)                    # (B, C, D)
            X[m] = x

        # eye_out = self.eye(eye)
        # gsr_out = self.gsr(gsr)
        # eeg_out = self.eeg(eeg)
        # ecg_out = self.ecg(ecg)
        # concat = torch.cat([eye_out, gsr_out, eeg_out, ecg_out], dim=1)
        concat = torch.cat([X[m] for m in self.modality_keys], dim=1)

        x = self.D1(concat)
        x = F.relu(self.BN1(x))
        x = self.D2(x)
        x = F.relu(self.BN2(x))
        x = self.drop(x)
        x = self.D3(x)
        x = F.relu(self.BN3(x))
        x = F.relu(self.D4(x))
        out = self.out_3(x)  # Softmax would be applied directly by CrossEntropyLoss, because labels=classes
        output = {}
        output['logits'] = out
        return output