import numpy as np
import pandas as pd

def simcheck(ar1,ar2):
    cleaned_str = ar1.replace("[", "").replace("]", "").replace("\n", "")
    numbers = list(map(float, cleaned_str.split()))
    ar1 = np.array(numbers).reshape(6, 7)
    cleaned_str = ar2.replace("[", "").replace("]", "").replace("\n", "")
    numbers = list(map(float, cleaned_str.split()))
    ar2 = np.array(numbers).reshape(6, 7)
    for i in range(6):
        for j in range(7):
            if ar1[i][j]==0 and ar2[i][j]!=0:
             a=10*i+j
             return a 
    return None
#Dataset=pd.read_csv('100k.csv')
#Dataset.columns=["result","state"]
#Dataset.inert(2,"move",0)
#for i in range(Dataset.shape[0]-1):
# board_state=Dataset['state'][i]
# Dataset.loc[i,'move']=simcheck(board_state,Dataset['state'][i+1])
#Dataset.to_csv("data.csv")
#print("______________________________________")
"""
dataset=pd.read_csv('data.csv')
dataset['move']=(dataset['move'])%10
dataset=pd.DataFrame(dataset).to_numpy()

for i in range(dataset.shape[0]):
   if dataset[i][3]!=dataset[i][3]:
     dataset[i][3]=0
   ar1=dataset[i][2]
   cleaned_str = ar1.replace("[", "").replace("]", "").replace("\n", "")
   numbers = list(map(float, cleaned_str.split()))
   ar1 = np.array(numbers).reshape(6, 7)
   dataset[i][2]=ar1
dataset=pd.DataFrame(dataset)
dataset.to_csv("Final_data.csv")
"""
dataset=pd.read_csv('Final_data.csv')
dataset=pd.DataFrame(dataset).to_numpy()
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
class Connect4Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extract board state and next move
        board_state = torch.tensor(self.data[idx][2], dtype=torch.float32)
        move=torch.tensor(self.data[idx][3], dtype=torch.int64)
        return board_state,move

train_loader = DataLoader(Connect4Dataset(dataset), batch_size=64, shuffle=True)
import torch.nn as nn
import torch.optim as optim

class Connect4Model(nn.Module):
    def __init__(self):
        super(Connect4Model, self).__init__()
        self.fc1 = nn.Linear(42, 128)  # 42 inputs for the 6x7 board
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 7)  # Output 7 possible moves (columns)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instantiate the model
model = Connect4Model()
# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(model, train_loader, epochs=20):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for board, move in train_loader:
            # Flatten the board state
            board = board.view(board.size(0), -1)
            
            # Forward pass
            outputs = model(board.float())
            loss = criterion(outputs, move)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}')

#train_model(model, train_loader)

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for board, move in test_loader:
            board = board.view(board.size(0), -1)
            outputs = model(board.float())
            _, predicted = torch.max(outputs.data, 1)
            total += move.size(0)
            correct += (predicted == move).sum().item()

    print(f'Accuracy: {100 * correct / total:.2f}%')

#evaluate_model(model, test_loader)

def play_connect4(model):
    board_state = np.zeros((6,7))  
    # Empty board, 6x7 = 42
    game_over = False
    
    while not game_over:
        # Player move (you can add code here for input validation)
        print("Your move (0-6):")
        player_move = int(input())
        for i in range(6):
          if board_state[i][player_move]==0:
           board_state[i][player_move] = 1
           break
        if wincheck(board_state,1):
         print("Player wins the game")
         break
        print(board_state)
        print("__________________________________________________")
        # Model's move
        board_tensor = torch.tensor(board_state).float().view(1, 42)
        model_output = model(board_tensor)
        _, ai_move = torch.max(model_output.data, 1)
        ai_move=ai_move.item()
        print(ai_move)
        for i in range(6):
          if board_state[i][ai_move]==0:
           board_state[i][ai_move] = 2
           break# Example of player move
        if(wincheck(board_state,2)): # Example of AI move
         print("AI wins the game")
         break
        # Print the board (you can add your own display logic)
        print(board_state)
        print("_________________________________________________")
    # Check horizontal locations for win
def wincheck(board_state,piece):
        for c in range(4):
         for r in range(6):
            if board_state[r][c] == piece and board_state[r][c+1] == piece and board_state[r][c+2] == piece and board_state[r][c+3] == piece:
                return True

    # Check vertical locations for win
        for c in range(7):
         for r in range(3):
            if board_state[r][c] == piece and board_state[r+1][c] == piece and board_state[r+2][c] == piece and board_state[r+3][c] == piece:
                return True

    # Check positively sloped diaganols
        for c in range(4):
         for r in range(3):
            if board_state[r][c] == piece and board_state[r+1][c+1] == piece and board_state[r+2][c+2] == piece and board_state[r+3][c+3] == piece:
                return True

    # Check negatively sloped diaganols
        for c in range(4):
         for r in range(3,6):
            if board_state[r][c] == piece and board_state[r-1][c+1] == piece and board_state[r-2][c+2] == piece and board_state[r-3][c+3] == piece:
                return True
         
play_connect4(model)  
# Add game-over check logic here
