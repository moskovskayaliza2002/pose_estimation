'''
Input layer parameters of proposed model are set as 1 × 25 × 2, where 1 shows that data for each frame will come and 25 × 2 represents 25 joint-location (X,Y) coordinates. Finally, the X, Y coordinates of extracted skeleton features are fed into the CNN-LSTM system for the classification process. CNN-LSTM architecture as shown in Figure 2 is designed in a way that it can work on skeleton features directly instead of generating heatmaps from skeleton features for the training of deep learning architecture. The proposed CNN-LSTM architecture consists of 12 layers starting with input layer for taking skeleton features as an input. The dimensions of an input layer are 1 × 25 × 2, where 1 shows that data are of a single frame and 25 × 2 are the X, Y coordinates of 25 joints locations. A time-distributed CNN layer with 16 filters of size 3 × 3 is used, and for feature extraction, ReLU activation is used on key points of each frame. CNNs are very good at pulling out spatial features that are not affected by scale or rotation. The CNN layer can extract spatial features and angles between the key points in a frame. Batch normalization is used to speed up convergence on the CNN output. The next layer is a dropout layer, which randomly drops some of the weights to avoid overfitting. The CNN output is then flattened and sent to the LSTM layer, which has 20 units and a unit forget bias of 0.5. LSTM is used to see how the features extracted by the CNN layer change over time. This takes advantage of the fact that video input comes in a certain order.
'''

from torch import nn
import torch


class CNN_LSTM_model(nn.Module):
    def __init__(self, p, num_classes):
        super(CNN_LSTM_model, self).__init__()
        self.convolutional_layer1 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size = 3),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(3),
            nn.ReLU(inplace=False))
        
        self.dropout = nn.Dropout(p=p)
        
        self.convolutional_layer2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size = 3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=False))
        
        self.flatten = nn.Flatten()
        
        self.relu = nn.ReLU(inplace=False)
        
        self.fc1 = nn.Linear(in_features=160, out_features=80)
        
        self.lstm = nn.LSTM(input_size=80, hidden_size=20, num_layers=1) #out = 20 units
        
        self.fc2 = nn.Linear(in_features=20, out_features=num_classes)
        
        self.softmax = nn.Softmax()
    
    def forward(self, x):
        print(f"x {x.shape}")
        first_step = self.convolutional_layer1(x)
        print(f"first_step {first_step.shape}")
        second_step = self.convolutional_layer2(self.dropout(first_step))
        print(f"second_step {second_step.shape}")
        third_step = self.flatten(self.dropout(second_step))
        print(f"third_step {third_step.shape}")
        fourth_step = self.relu(self.fc1(third_step))
        print(f"fourth_step {fourth_step.shape}")
        fifth_step, (hn, cn) = self.lstm(fourth_step)
        print(f"fifth_step {fifth_step.shape}")
        sixth_step = self.relu(self.fc2(fifth_step))
        print(f"sixth_step {sixth_step.shape}")
        output = self.softmax(sixth_step)
        print(output)
        

if __name__ == "__main__":
    # in keras input size [1, 25, 2] in pytorch [1, 2, 25]
    tensor = torch.randint(3, 10, (1, 25, 2))
    correct_pytorch_version = tensor.transpose(2, 1)
    model = CNN_LSTM_model(0.2, 5)
    correct_pytorch_version = correct_pytorch_version.float() 
    #tensor = tensor.view(len(tensor), 1, -1)
    print(correct_pytorch_version.shape)
    output = model(correct_pytorch_version)
    
    
