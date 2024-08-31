import torch
from modelTraining import ChessBoardModel
import torchvision.transforms as T
from PIL import Image

PATH = "./"
IMAGE_PATH = "./Images/train/RRr5-2r3b1-8-2b4p-pK3P2-3p4-8-1n1k2n1.jpeg"

model = ChessBoardModel()
model.load_state_dict(torch.load((PATH + "model.pth"), weights_only=False))
model.eval() # Set to evaluation mode to reduce possible errors


TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(), # standard type for PyTorch
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = Image.open(IMAGE_PATH)
image = TRANSFORM(image).unsqueeze(0) # Add a dimension to the tensor to make it 4D

def parse_fen (fenstring):
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

with torch.no_grad():

    pred = model(image) # Forward pass image through model trained previously
    pred_out, pred_index = torch.max(pred.permute(0,3,1,2), 1) # Get the index of the highest value in the output tensor

print(pred_index,"\n\n\n") 
print (parse_fen("RRr5-2r3b1-8-2b4p-pK3P2-3p4-8-1n1k2n1")) 