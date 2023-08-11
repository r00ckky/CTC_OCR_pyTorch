import torch
from torch.utils import data
from data_engine import WritingDataGen
from model import CRNN

model = CRNN()

TRAIN_LOSS_LIST = []

n_epochs = 10

TRAIN_DATA = WritingDataGen(img_dataframe=ds_train, tar_size=(64,256), img_dir=TRAIN_DIR, token_dict=token_dict)
TRAIN_LOADER = data.DataLoader(TRAIN_DATA, batch_size=4, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

model.train(True)
model.to('cuda')

for epoch in range(n_epochs):
    print(f'EPOCHS: {epoch+1}', end=" ")
    train_loss = 0
    model.train(True)
    model.to('cuda')
    for img, target in TRAIN_LOADER:
        img = img.to('cuda')
        optimizer.zero_grad()
        out, loss = model(img, target)
        loss.backward()
        optimizer.step()
        train_loss+=loss.to('cpu').detach().numpy()
    
    train_loss = train_loss/len(TRAIN_LOADER)
    TRAIN_LOSS_LIST.append(train_loss)
    print(f"|Train Loss: {train_loss}")