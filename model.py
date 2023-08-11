import torch 
from torch import nn
from torch.nn import functional as F

class CRNN(nn.Module):
    def __init__(self):
        super(CRNN, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 16, kernel_size=(3,3), padding='same')
        self.conv1_2 = nn.Conv2d(16, 64, kernel_size=(3,3), padding='same')
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2))
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=(3,3), padding='same')
        self.conv2_2 = nn.Conv2d(128, 256, kernel_size=(3,3), padding='same')
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2))
        self.linear1 = nn.Linear(4096, 256)
        self.drop = nn.Dropout(0.3)
        self.lstm = nn.LSTM(input_size=256, hidden_size=32, num_layers=2, bidirectional=True)
        self.out = nn.Linear(64, 24)

    def forward(self, image, targets=None):
        bs, ch, hi, wd = image.shape
        x = F.relu(self.conv1_1(image))
        x = F.relu(self.conv1_2(x))
        x = self.maxpool1(x)  # 1, 64, 32, 128
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.maxpool2(x) # 1, 256, 16, 64

        x = x.permute(0, 3, 1, 2) # 1, 64, 256, 16
        x = x.view(bs, x.size(1), -1) # 1, 64, 4096

        x = self.linear1(x)
        x = self.drop(x) # 4, 64, 256
        x, _ = self.lstm(x) # 4, 64, 64
        x = self.out(x) # 4,64,24
        x = x.permute(1, 0, 2)
        if targets is not None:
            log_softmax_vals = F.log_softmax(x, 2)
            loss = nn.CTCLoss(blank=0)(
                log_softmax_vals, targets[0], targets[1], targets[2]
            )
            return x, loss
        return x, None