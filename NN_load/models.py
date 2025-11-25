
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Net"]

class Net(nn.Module):
    def __init__(self, out_hw=(65,65),
                 H1=256, H2=512, H3=1024, H4=2048, dropout=0.05):
        super().__init__()
        self.H, self.W = out_hw
        self.out_dim = self.H * self.W
        in_dim = 16 + 2  # IPFs + [Ip, beta_p]

        self.fc1 = nn.Linear(in_dim, H1); self.ln1 = nn.LayerNorm(H1)
        self.fc2 = nn.Linear(H1, H2);     self.ln2 = nn.LayerNorm(H2)
        self.fc3 = nn.Linear(H2, H3);     self.ln3 = nn.LayerNorm(H3)
        self.fc4 = nn.Linear(H3, H4);     self.ln4 = nn.LayerNorm(H4)
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(H4, 4096), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4096, self.out_dim)
        )
        for m in [self.fc1, self.fc2, self.fc3, self.fc4]:
            nn.init.kaiming_uniform_(m.weight, a=0.0); nn.init.zeros_(m.bias)

    def forward(self, ipfs16, cond2):     # ipfs16:(B,16), cond2:(B,2)
        x = torch.cat([ipfs16, cond2], dim=1)       # (B,18)
        h = F.relu(self.ln1(self.fc1(x)))
        h = F.relu(self.ln2(self.fc2(h)))
        h = F.relu(self.ln3(self.fc3(h)))
        h = F.relu(self.ln4(self.fc4(h)))
        y = self.head(h)                             # (B,4225)
        return y.view(-1, 1, self.H, self.W)        # (B,1,65,65)
