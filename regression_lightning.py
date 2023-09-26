import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split, DataLoader
from transformers import AutoModel, AutoTokenizer
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

EPOCHS = 100
SEED = 2023
BERT = 'DeepPavlov/distilrubert-tiny-cased-conversational'
MAX_LENGTH = 200
BATCH_SIZE = 4

input_col = 'INPUT:text'
outpul_cols = ['OUTPUT:pleasure', 'OUTPUT:arousal', 'OUTPUT:dominance']
output_features = len(outpul_cols)
data_url = "https://storage.yandexcloud.net/nlp-dataset-bucket-1/pad/toloka_raw_pad-04-2023.tsv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PADRegressionDataset(Dataset):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(BERT)

        df = pd.read_csv(data_url, sep='\t', usecols=[input_col, *outpul_cols], index_col=False, header=0)
        df = df.dropna(how='all')
        self.df = df

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[[index]]
        text = row[input_col].values[0]
        scale_values = row[outpul_cols].values[0]

        tokens = self.tokenizer.encode(text, padding='max_length', add_special_tokens=True, max_length=MAX_LENGTH, truncation=True)

        x = torch.tensor(tokens)
        y = torch.tensor(scale_values, dtype=torch.float)
        return x, y


class RegressionBert(pl.LightningModule):
    def __init__(self, freeze_bert=True):
        super(RegressionBert, self).__init__()
        self.bert = AutoModel.from_pretrained(BERT, num_labels=1)
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
        self.fc0 = torch.nn.Linear(768, 2048)
        self.fc1 = torch.nn.Linear(2048, 512)
        self.fc2 = torch.nn.Linear(512, 128)
        self.fc3 = torch.nn.Linear(128, 3)

        self.lrelu = torch.nn.LeakyReLU()

        self.train_loss = torch.nn.MSELoss()
        self.val_loss = torch.nn.L1Loss()

    def forward(self, x, att=None):
        x = self.bert(x, attention_mask=att)[0]
        x = torch.mean(x, dim=1)
        #x = torch[:,0,:]
        x = self.lrelu(self.fc0(x))
        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc2(x))
        x = self.fc3(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.00001)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        inputs, values = train_batch
        outputs = self.forward(inputs)
        loss = self.train_loss(outputs, values)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, train_batch, batch_idx):
        inputs, values = train_batch
        outputs = self.forward(inputs)
        loss = self.val_loss(outputs, values)
        self.log('val_loss', loss)

        return loss

dataset = PADRegressionDataset()
data_train, data_val = random_split(dataset, [150, 42])

train_loader = DataLoader(data_train, batch_size=BATCH_SIZE)
val_loader = DataLoader(data_val, batch_size=BATCH_SIZE)

if __name__ == '__main__':
    model = RegressionBert()

    # training
    early_stopping = EarlyStopping(monitor="val_loss", min_delta=0.05, mode='min')
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="models",
        filename="regressionregression-bert-{val_loss:.4f}",
        save_top_k=2,
        mode="min",
    )

    logger = TensorBoardLogger('logs')
    trainer = pl.Trainer(log_every_n_steps=15, logger=logger, callbacks=[checkpoint_callback])
    trainer.fit(model, train_loader, val_loader)