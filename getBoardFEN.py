from imageNormalization import detect_chessboard, split_chessboard, ChessDataset
import chess
from modelCreation import ChessPieceModel
import torch
import torchvision.transforms as T


def getSquares(image_path):
    # Detect the chessboard in the image
    ret, corners, image = detect_chessboard(image_path)
    
    if not ret:
        return None

    # Split the chessboard into 64 squares
    path_to_squares = split_chessboard(image, corners)
    

    TRANSFORM = T.Compose([
        T.Resize((28, 28)),
        T.ToTensor() # standard type for PyTorch
        #T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    squares = ChessDataset(path_to_squares, transform=TRANSFORM, shuffle = False)  # Convert the list of squares to a PyTorch dataset
    
    return squares
    


def getFen(path_to_model,squares):
    
    # Load my pre-trained model
    model = ChessPieceModel()
    model.load_state_dict(torch.load(path_to_model, weights_only=False))
    model.eval() # Set to evaluation mode to reduce possible errors
    
    predictions = []
    for image, _ in squares:
    
        image = image.unsqueeze(0)
    
        with torch.no_grad():
    
            pred = model(image) # Forward pass image through model trained previously
            pred_out, pred_index = torch.max(pred.data, 1) # Get the index of the highest value in the output tensor
            
            predictions.append(pred_index.item())
    print (pred_out)
    print(pred_index)
    
    return convertToFen(predictions)
            
            
def convertToFen(predictions):
    # Initialize an empty FEN string
    fen = ""
    
    # Map the piece indices to FEN characters
    FEN_MAP = {
        1: 'P', 2: 'R', 3: 'N', 4: 'B', 5: 'Q', 6: 'K',  # White pieces
        7: 'p', 8: 'r', 9: 'n', 10: 'b', 11: 'q', 12: 'k', # Black pieces
        0: ' '  # Empty square
    }
    
    # Convert the piece indices to FEN characters
    for i, piece in enumerate(predictions):
        if i % 8 == 0 and i != 0:
            fen += "/"
        if piece == 0:
            if not fen[-1].isdigit():
                fen += "1"
            else:
                fen = fen[:-1] + str(int(fen[-1]) + 1)
        else:
            fen += FEN_MAP[piece]
    
    # Return the FEN string
    return fen


if __name__ == "__main__":
    import random
    test_list = [random.randint(0,12) for _ in range(56)]
    test_list += [0]*5
    test_list += [random.randint(0,12) for _ in range(3)]
    print(test_list)
    fen = convertToFen(test_list)
    
    print(fen)
            
            
            
            

    
        
        

    

