import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import glob

class ChessDataset(Dataset):
    def __init__(self, path, transform=None):
        self.images = [i for i in glob.glob(path + "/*.jpeg")]
        self.labels = [l.split("\\")[-1].replace(".jpeg", "") for l in self.images]
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")

        if self.transform:
            img = self.transform(img)

        #print (self.labels[idx])
        label = self.parse_fen(self.labels[idx])

        return img, label 
    
    def parse_fen (self, fenstring):
        FEN_MAP = {
            'P': 1, 'R': 2, 'N': 3, 'B': 4, 'Q': 5, 'K': 6,  # White pieces
            'p': 7, 'r': 8, 'n': 9, 'b': 10, 'q': 11, 'k': 12, # Black pieces
            ' ': 0  # Empty square
        }
        board = []

        rows = fenstring.split('-')

        for row in rows:
            temp = []
            for x in row:
                if x in FEN_MAP:
                    temp.append(FEN_MAP[x])
                else:
                    temp += [0] * int(x)

            board.append(temp)
        
        return torch.tensor(board, dtype=torch.int64)
    



if __name__ == "__main__":
    IMAGES_PATH = ".\Images\Train"
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ChessDataset(IMAGES_PATH, transform=transform)

    print(dataset[0])