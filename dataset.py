



from torch.utils.data import Dataset


class MsDataset(Dataset):
     def __init__(self,signals,labels):
         self.signals=signals
         self.labels=labels

     def __len__(self):
        return len(self.signals)

     def __getitem__(self, idx):
        signal,label= self.signals[idx], self.labels[idx]
        return signal,label