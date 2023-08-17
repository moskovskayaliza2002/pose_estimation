'''
Input layer parameters of proposed model are set as 1 × 25 × 2, where 1 shows that data for each frame will come and 25 × 2 represents 25 joint-location (X,Y) coordinates. Finally, the X, Y coordinates of extracted skeleton features are fed into the CNN-LSTM system for the classification process. CNN-LSTM architecture as shown in Figure 2 is designed in a way that it can work on skeleton features directly instead of generating heatmaps from skeleton features for the training of deep learning architecture. The proposed CNN-LSTM architecture consists of 12 layers starting with input layer for taking skeleton features as an input. The dimensions of an input layer are 1 × 25 × 2, where 1 shows that data are of a single frame and 25 × 2 are the X, Y coordinates of 25 joints locations. A time-distributed CNN layer with 16 filters of size 3 × 3 is used, and for feature extraction, ReLU activation is used on key points of each frame. CNNs are very good at pulling out spatial features that are not affected by scale or rotation. The CNN layer can extract spatial features and angles between the key points in a frame. Batch normalization is used to speed up convergence on the CNN output. The next layer is a dropout layer, which randomly drops some of the weights to avoid overfitting. The CNN output is then flattened and sent to the LSTM layer, which has 20 units and a unit forget bias of 0.5. LSTM is used to see how the features extracted by the CNN layer change over time. This takes advantage of the fact that video input comes in a certain order.
'''

from torch import nn
import torch.optim as optim
import torch
from pytorch_openpose_body_25.src import torch_openpose,util
import cv2
import argparse
import numpy as np
from sklearn.metrics import classification_report

NAMES = ['Nose', 'Mid_shoulder', 'Right_shoulder', 'Right_elbow', 'Right_hand', 'Left_shoulder', 'Left_elbow', 'Left_hand', 'Mid_hip', 'Right_hip', 'Right_knee', 'Right_ankle', 'Left_hip', 'Left_knee', 'Left_ankle', 'Right_eye', 'Left_eye', 'Right_ear', 'Left_ear', 'Left_big_toe', 'Left_small_toe', 'Left_heel', 'Right_big_toe', 'Right_small_toe', 'Right_heel']

class ClassDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.keypoints_files = sorted(os.listdir(os.path.join(root, "keypoints")))
        self.pose_class_files = sorted(os.listdir(os.path.join(root, "pose_class")))
    
    def __getitem__(self, idx):
        keypoints_path = os.path.join(self.root, "keypoints", self.keypoints_files[idx])
        pose_class_files = os.path.join(self.root, "pose_class", self.pose_class_files[idx])
        
        keypoints_original = []
        labels_original = []
        
        keypoints_original = read_kps(keypoints_path)
                
        with open(pose_class_files) as f:
            for i in f:
                label = map(int, i[:-1].split(' '))
                labels_original.append(label)
        
        x = torch.as_tensor(keypoints, dtype=torch.float32)
        y = torch.as_tensor(labels_original, dtype=torch.int64)
        
        return x, y
        
            
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
        
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # x shape [Batch, 2, 25]
        first_step = self.convolutional_layer1(x) # first_step shape [Batch, 16, 7]
        second_step = self.convolutional_layer2(self.dropout(first_step)) # second_step shape [Batch, 32, 5]
        third_step = self.flatten(self.dropout(second_step)) # third_step shape [Batch, 160]
        fourth_step = self.relu(self.fc1(third_step)) # fourth_step shape [Batch, 80]
        fifth_step, (hn, cn) = self.lstm(fourth_step) # fifth_step shape [Batch, 20]
        sixth_step = self.relu(self.fc2(fifth_step)) # fifth_step shape [Batch, num_classes]
        output = self.softmax(sixth_step)
        return output
    
    
def plot_loss_acc(H, path):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["val_loss"], label="val_loss")
    plt.plot(H["train_acc"], label="train_acc")
    plt.plot(H["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="best")
    plt.savefig(path)
    
    
def train(root, num_classes = 5):
    INIT_LR = 1e-3
    BATCH_SIZE = 64
    EPOCHS = 10
    TRAIN_SPLIT = 0.75
    VAL_SPLIT = 1 - TRAIN_SPLIT
    DROPOUT_P = 0.2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=INIT_LR, momentum=0.9)
    model = CNN_LSTM_model(DROPOUT_P, num_classes)
    print("[INFO] started training the model")
    startTime = time.time()
    
    # FOLDERS
    KEYPOINTS_FOLDER_TRAIN = root + '/Train_data'
    KEYPOINTS_FOLDER_TEST = root + '/Test_data'
    SAVING_WEIGHTS_PATH = root + '/cnn_lstm_models/'
    if not os.path.exists(SAVING_WEIGHTS_PATH):
        os.mkdir(SAVING_WEIGHTS_PATH)
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    dir = os.path.join(SAVING_WEIGHTS_PATH, time_str)
    if not os.path.exists(dir):
        os.mkdir(dir)
    SAVING_WEIGHTS_PATH += time_str
    
    trainData = ClassDataset(KEYPOINTS_FOLDER_TRAIN)
    testData = ClassDataset(KEYPOINTS_FOLDER_TEST)
    numTrainSamples = int(len(trainData) * TRAIN_SPLIT)
    numValSamples = int(len(trainData) * VAL_SPLIT)
    (trainData, valData) = random_split(trainData,[numTrainSamples, numValSamples], generator=torch.Generator().manual_seed(42))
    
    trainDataLoader = DataLoader(trainData, shuffle=True, batch_size=BATCH_SIZE)
    valDataLoader = DataLoader(valData, batch_size=BATCH_SIZE)
    testDataLoader = DataLoader(testData, batch_size=BATCH_SIZE)
    
    # calculate steps per epoch for training and validation set
    trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
    valSteps = len(valDataLoader.dataset) // BATCH_SIZE
    
    for e in range(0, EPOCHS):
        model.train()
        totalTrainLoss = 0
        totalValLoss = 0
        trainCorrect = 0
        valCorrect = 0
        for (x, y) in trainDataLoader:
            (x, y) = (x.to(device), y.to(device))
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totalTrainLoss += loss
            trainCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()
            with torch.no_grad():
                model.eval()
                for (x, y) in valDataLoader:
                    (x, y) = (x.to(device), y.to(device))
                    pred = model(x)
                    totalValLoss += criterion(pred, y)
                    valCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()
            avgTrainLoss = totalTrainLoss / trainSteps
            avgValLoss = totalValLoss / valSteps
            trainCorrect = trainCorrect / len(trainDataLoader.dataset)
            valCorrect = valCorrect / len(valDataLoader.dataset)
            
            if avgValLoss < min_loss:
                min_loss = avgValLoss
                torch.save(model.state_dict(), SAVING_WEIGHTS_PATH + '/best_weights.pth')
                
            H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
            H["train_acc"].append(trainCorrect)
            H["val_loss"].append(avgValLoss.cpu().detach().numpy())
            H["val_acc"].append(valCorrect)
            
    torch.save(model.state_dict(), SAVING_WEIGHTS_PATH + '/full_weights.pth')
            
    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))
    with torch.no_grad():
        model.eval()
        preds = []
        for (x, y) in testDataLoader:
            x = x.to(device)
            pred = model(x)
            preds.extend(pred.argmax(axis=1).cpu().numpy())
    print(classification_report(testData.targets.cpu().numpy(), np.array(preds), target_names=testData.classes))
    
    plot_loss_acc(H, SAVING_WEIGHTS_PATH + '/loss_acc_plot.png')
    

def write_kps(kps, filename):
    # Записывает список координат в файл
    str_list = []
    for i in kps:
        s = ' '.join(map(str, i)) + "\n"
        str_list.append(s)
    with open(filename, 'w') as file:
        file.writelines(str_list)
        file.close()
        
        
def create_keypoint_data(image_data_path, keypoint_path):
    TRAIN_IMAGES_PATH = root + '/Train_data' + '/images'
    TEST_IMAGES_PATH = root + '/Test_data' + '/images'
    TRAIN_KEYPOINTS_PATH = root + '/Train_data' + '/keypoints'
    TEST_KEYPOINTS_PATH = root + '/Test_data' + '/keypoints'
    
    model_poses = torch_openpose.torch_openpose('body_25')
    model_poses.eval()
    
    for f in os.scandir(image_data_path):
        if f.is_file() and f.path.split('.')[-1].lower() == 'png':
            number = int(f.path[f.path.rfind('e') + 1 : f.path.rfind('.')])
            img = cv2.imread(f.path)
            output = np.array(model_poses(img))
            state_poses = output[:,:,:2] # shape (1, 25, 2) [batch, num_kps, x y]
            write_kps(np.reshape(state_poses.flatten(), (-1, 50)), keypoint_path + '/keypoint' + str(number) + '.txt')
            
    
def read_kps(path):
    # Считывает ключевые точки
    kps_original = []
    with open(path) as f:
        for i in f:
            kps = list(map(int, i[:-1].split(' ')))
            kps = torch.tensor(np.array(kps).reshape((25, 2))).transpose(2, 1).tolist()
            kps_original.append(kps)
    return kps_original
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', nargs='?', default="pytorch_openpose_body_25/images/timg.jpeg", help="Specify the path to data root folder", type=str)
    args = parser.parse_args()
    root = args.root
    
    TRAIN_KEYPOINTS_PATH = root + '/Train_data' + '/keypoints'
    TEST_KEYPOINTS_PATH = root + '/Test_data' + '/keypoints'
    TRAIN_IMAGES_PATH = root + '/Train_data' + '/images'
    TEST_IMAGES_PATH = root + '/Test_data' + '/images'
    
    #create keypoints labels with openpose
    if not os.path.exists(TRAIN_KEYPOINTS_PATH):
        os.mkdir(TRAIN_KEYPOINTS_PATH)
        create_keypoint_data(TRAIN_IMAGES_PATH, TRAIN_KEYPOINTS_PATH)
    if not os.path.exists(TEST_KEYPOINTS_PATH):
        os.mkdir(TEST_KEYPOINTS_PATH)
        create_keypoint_data(TEST_IMAGES_PATH, TEST_KEYPOINTS_PATH)
        
    #train LSTM model
    train(root)
    '''
    #____________testing______________ 
    model_poses = torch_openpose.torch_openpose('body_25')
    img = cv2.imread(args.test_data_path)
    output = np.array(model_poses(img)) # shape (1, 25, 3) [batch, num_kps, x y conf]
    state_poses = torch.tensor(output[:,:,:2]) # shape (1, 25, 2) [batch, num_kps, x y]
    correct_pytorch_version = state_poses.transpose(2, 1)
    model_estimate = CNN_LSTM_model(0.2, 5)
    dynamic_poses = model_estimate(correct_pytorch_version.float())
    print(dynamic_poses.shape)
    '''
    
