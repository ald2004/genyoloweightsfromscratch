import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5) #10 x  10 x  16
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train():
    net = Net().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                            shuffle=True, num_workers=2)


    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs.cuda())
            loss = criterion(outputs, labels.cuda())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

def conver_2_yolo(pymodel='./cifar_net.pth'):
    #define MAJOR_VERSION 0
    #define MINOR_VERSION 2
    #define PATCH_VERSION 5
    LAYER_TOBE_CONVERTED=['conv','fc']
    LAYER_WEIGHT_TYPE=['.bias','.weight']
    CPU_DEVICE=torch.device("cpu")

    yversionarray=np.array([0,2,5],dtype=np.int32)
    yiseen=np.array([1920000],dtype=np.int64)

    # model=Net()
    statedict=torch.load(pymodel)
    metadata=getattr(statedict,"_metadata",None)
    # model.load_state_dict(statedict)
    with open('test_np_w_yolo_r.npy','wb') as fid:
        fid.write(yversionarray.tobytes())
        fid.write(yiseen.tobytes())
        # ------------------------------------
        # ---------  configuration file   ----
        # ----------------------------------->
        #
        # ---------     ------------    ---------
        # --conv1--     --shortcut2-    --conv3--
        # ---------     ------------    ---------
        #
        # ----------------------------------->
        # ---------   weights  file  ---------
        # ----------------------------------->
        #
        #    conv1                    conv2
        #  bn_biases    bn_weights  bn_running_mean  bn_running_var  conv_weights if conv with bn
        #  conv_biases  conv_weights else without bn
        state_dict = statedict.copy()
        if metadata is not None:
            state_dict._metadata = metadata
        
        for idx,k in enumerate(metadata):
            if k:
                for ltobe in LAYER_TOBE_CONVERTED:
                    if ltobe in k:
                        for weighttype in LAYER_WEIGHT_TYPE:
                            print(f"\t {idx}, -> {state_dict[k+weighttype].shape}") 
                            curr_weight=state_dict[k+weighttype].to(CPU_DEVICE)
                            xxnumwirte=curr_weight.numel()
                            curr_weight=np.asarray(curr_weight.numpy(),dtype=np.float32)
                            numwrite=fid.write(curr_weight.tobytes())
                            print(f"\t\t torch number :{xxnumwirte*4}, file write size:{numwrite}")
                            assert(not (xxnumwirte*4-numwrite))
            else:
                print(idx,metadata[k])

            
            # conv1.weight
            # conv1.bias
            # conv2.weight
            # conv2.bias
            # fc1.weight
            # fc1.bias
            # fc2.weight
            # fc2.bias
            # fc3.weight
            # fc3.bias
    

conver _2_yolo()
# train()
