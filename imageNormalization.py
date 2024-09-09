import cv2
import os
import glob
import shutil
import numpy as np
import time
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image


def decode_fen(fen):
    # Decode the FEN string
    board = []
    for row in fen.split('-'):
        board_row = []
        for char in row:
            if char == '/':
                continue
            if char.isdigit():
                for _ in range(int(char)):
                    board_row.append('s') # s for space
            else:
                board_row.append(char)
        board.append(board_row)
    return board

def detect_chessboard(image_path):
    # Read the image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect the corners of the chessboard with improved accuracy
    ret, corners = cv2.findChessboardCornersSB(gray, (7, 7), cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_LARGER)
    
    return ret, corners, gray

# Split the chessboard into 64 squares
def split_chessboard(image, corners):
    # Resize the image to fit the chessboard

    # Get the dimensions of the chessboard inner corners
    new_width = int(corners[-1][0][0] - corners[0][0][0])
    new_height = int(corners[-1][0][1] - corners[0][0][1])

    # Use it to calculate the square dimensions and round up always
    square_height = int(new_height / 6 + 0.5) 
    square_width = int(new_width / 6 + 0.5)

    # Crop the image to the chessboard

    # Get the top left x and y coordinates
    t_l_y = 0 if int(corners[0][0][1])-square_width < 0 else int(corners[0][0][1])-square_width
    t_l_x = 0 if int(corners[0][0][0])-square_height < 0 else int(corners[0][0][0])-square_height

    # Get the bottom right x and y coordinates
    b_r_y = image.shape[1] if int(corners[-1][0][1])+square_width > image.shape[0] else int(corners[-1][0][1])+square_width
    b_r_x = image.shape[0] if int(corners[-1][0][0])+square_height > image.shape[1] else int(corners[-1][0][0])+square_height

    croped_image = image[t_l_y:b_r_y, t_l_x:b_r_x] 

    # Split the image into 64 squares

    # Process each square
    for i in range(8):
        for j in range(8):
            square = croped_image[i * square_height:(i + 1) * square_height,
                            j * square_width:(j + 1) * square_width] 
            square_name = "temp" + str(i) + str(j)
            # save the square to a folder
            square_folder = '.\\temp'
            if not os.path.exists(square_folder):
                os.makedirs(square_folder)
            filename = os.path.join(square_folder, f'{square_name}.jpg')
            try:
                cv2.imwrite(filename, square)
            except:
                print(f"Failed to save {filename}")
                
    return square_folder
            

def detect_chessboard_split(indexstart = 0, indexend = -1):

    start_time = time.time()
    # Loop through all the images in the folder
    unique_id = 0
    failed_count = 0
    count = -1
    for image_path in glob.glob(os.path.join(image_folder, '*.jpeg'))[indexstart:indexend]:
        # Read the image
        if count ==0:
            image_path = "Images\\rnbqkbnr-pppppppp-8-8-8-8-PPPPPPPP-RNBQKBNR.jpeg"
            count += 1

        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))

        
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect the corners of the chessboard with improved accuracy
        ret, corners = cv2.findChessboardCornersSB(gray, (7, 7), cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_LARGER)

        if ret:
            unique_id += 1
            image = gray.copy()
            
            # Resize the image to fit the chessboard

            # Get the dimensions of the chessboard inner corners
            new_width = int(corners[-1][0][0] - corners[0][0][0])
            new_height = int(corners[-1][0][1] - corners[0][0][1])

            # Use it to calculate the square dimensions and round up always
            square_height = int(new_height / 6 + 0.5) 
            square_width = int(new_width / 6 + 0.5)


            # Crop the image to the chessboard

            # Get the top left x and y coordinates
            t_l_y = 0 if int(corners[0][0][1])-square_width < 0 else int(corners[0][0][1])-square_width
            t_l_x = 0 if int(corners[0][0][0])-square_height < 0 else int(corners[0][0][0])-square_height

            # Get the bottom right x and y coordinates
            b_r_y = image.shape[1] if int(corners[-1][0][1])+square_width > image.shape[0] else int(corners[-1][0][1])+square_width
            b_r_x = image.shape[0] if int(corners[-1][0][0])+square_height > image.shape[1] else int(corners[-1][0][0])+square_height

            croped_image = image[t_l_y:b_r_y, t_l_x:b_r_x] 

            # Split the image into 64 squares

            board = decode_fen(image_path.split('\\')[-1].replace('.jpg', ''))

            # Process each square
            for i in range(8):
                for j in range(8):
                    square = croped_image[i * square_height:(i + 1) * square_height,
                                    j * square_width:(j + 1) * square_width]
                    
                    square_name = board[i][j]
                    # save the square to a folder
                    square_folder = os.path.join('./Images_Split', type_folder)
                    if not os.path.exists(square_folder):
                        os.makedirs(square_folder)
                    

                    filename = os.path.join(square_folder, f'{square_name}_{i}_{j}_{unique_id}.jpg')
                    try:
                        cv2.imwrite(filename, square)
                    except:
                        print(f"Failed to save {filename}")

                    

        else:
            failed_count += 1

            print(f"Chessboard not found in {image_path}, failed count: {failed_count}")
            # Create a directory for failed images
            failed_folder = os.path.join('./Failed_Images', type_folder)
            if not os.path.exists(failed_folder):
                os.makedirs(failed_folder)
            
            # Copy the failed image to the failed folder
            failed_image_path = os.path.join(failed_folder, os.path.basename(image_path))
            shutil.copy(image_path, failed_image_path)

    end_time = time.time()
    time_difference = end_time - start_time
    print(f"Time taken: {time_difference} seconds")


# Count the piece types
def count_pieces():
    piece_count = {
        'P': 0, 'R': 0, 'N': 0, 'B': 0, 'Q': 0, 'K': 0,
        'p': 0, 'r': 0, 'n': 0, 'b': 0, 'q': 0, 'k': 0, 's': 0 # s for space
    }

    for image_path in glob.glob(os.path.join('./Images_Split', type_folder, '*.jpg')):
        piece_name = os.path.basename(image_path).split('_')[0]
        piece_count[piece_name] += 1


    for piece, count in piece_count.items():
        if piece == 's':
            print(f"Empty square: {count}")
        elif piece.isupper():
            print(f"W{piece}: {count}")
        else:
            print(f"B{piece}: {count}")

# Sort the images into folders based on the piece type
def sort_images():
    for image_path in glob.glob(os.path.join('./Images_Split', type_folder, '*.jpg')):
        piece_name = os.path.basename(image_path).split('_')[0]
        piece_folder = os.path.join('./Images_Split', type_folder, piece_name)
        if not os.path.exists(piece_folder):
            os.makedirs(piece_folder)
        
        shutil.move(image_path, os.path.join(piece_folder, os.path.basename(image_path)))

# Normalize the images
# and format them in a class model
class ChessDataset(Dataset):
    def __init__(self, path, transform=None, shuffle=True):
        self.path = path
        self.transform = transform

        self.FEN_MAP = {
            'P': 1, 'R': 2, 'N': 3, 'B': 4, 'Q': 5, 'K': 6,  # White pieces
            'p': 7, 'r': 8, 'n': 9, 'b': 10, 'q': 11, 'k': 12, # Black pieces
            's': 0  # Empty square
        }
             
        self.images = []
        self.labels = []
        self.getFiles()
        if shuffle:
            self.randomizeListOrder()
    
    def getFiles(self):
        folders = glob.glob(self.path + '\\*')
        
        if len (folders) <= 8:
            for folder in folders:
                folder_name = self.FEN_MAP[folder.split('\\')[-1]]
                files = glob.glob(folder + '\\*')

                if folder_name == 0:
                    files = files[:len(files)//55]

                
                self.images.extend(files)
                self.labels.extend(folder_name for _ in range(len(files)))
        else:
            files = folders
            self.images.extend(files)
            self.labels.extend(0 for _ in range(len(files)))
    
    # Randomize the order of the images
    def randomizeListOrder(self):
        combined = list(zip(self.images, self.labels))
        np.random.shuffle(combined)
        self.images, self.labels = zip(*combined)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):

        if not isinstance(idx, int): # if given a Slice and not just one index
            arr = []
            for i in range(idx.start, idx.stop, idx.step if idx.step else 1):
                try:
                    img = Image.open(self.images[i])
                except:
                    img = self.images[i]
                if self.transform:
                    img = self.transform(img)
                label = self.labels[i]
                arr.append((img, label))
            return arr
        try:
            img = Image.open(self.images[idx])
        except:
            img = self.images[idx]

        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]

        return img, torch.tensor(label, dtype=torch.int64) 

if __name__ == '__main__':
    # Path to the image folder
    type_folder = 'train'
    image_folder = './Images/' + type_folder

    detect_chessboard_split(20000,80000)
    count_pieces()
    sort_images()

    """
    IMAGES_PATH = ".\\Images_Split\\train"
    transform = T.Compose([
        T.Resize((28, 28)),
        T.ToTensor()
        #T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ChessDataset(IMAGES_PATH, transform=transform)

    print(dataset[0][1])
    """

    
    