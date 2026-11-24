
import torch
from torch import nn
from code import config
class MTNet(nn.Module):
    def __init__(self,d):
        super().__init__(); self.encoder=nn.Sequential(nn.Linear(d,config.HIDDEN[0]),nn.ReLU(),nn.Dropout(config.DROPOUT),nn.Linear(config.HIDDEN[0],config.HIDDEN[1]),nn.ReLU(),nn.Dropout(config.DROPOUT)); self.heads=nn.ModuleDict({e:nn.Linear(config.HIDDEN[1],1) for e in config.ENDPOINTS})
    def forward(self,x): z=self.encoder(x); return {e:self.heads[e](z).squeeze(1) for e in config.ENDPOINTS}
def load_model():
    import json
    p=config.DNN_CONFIG_FILE
    if not config.DNN_MODEL_FILE.exists() or not config.DNN_SCALER_FILE.exists() or not p.exists(): raise FileNotFoundError("Trained DNN, scaler, or config missing. Run 04_multitask_dnn.py.")
    d=json.load(open(p)); m=MTNet(d["input_dim"]); m.load_state_dict(torch.load(config.DNN_MODEL_FILE,map_location="cpu",weights_only=True)); m.eval(); return m
