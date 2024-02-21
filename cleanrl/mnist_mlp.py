import os
import random
import time
import numpy as np # 선형대수 모듈
import matplotlib.pyplot as plt # 시각화 모듈
import torch # 파이토치
import torch.nn as nn # PyTorch의 모듈을 모아놓은 것. from~~이 아닌 저렇게 임포트를 하는 것이 거의 관습이라고 한다.
import torch.nn.functional as F # torch.nn 중에서 자주 쓰는 함수를 F로 임포트.
import torch.nn.init as init # 초기화 관련 모듈 
import torchvision # TorchVision 임포트
from torchvision import transforms, datasets # 데이터를 다루기 위한 TorchVision 내의 Transforms와 datasets를 따로 임포트
from dataclasses import dataclass
import tyro
from torch.utils.tensorboard import SummaryWriter

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    c_bias: int = 1000
    c_weight: int = 1000
    number_first_hidden: int = 512
    number_second_hidden: int = 256
    number_samples: int = 2000
    torch_deterministic: bool = True
    seed: int =2
    """seed of the experiment"""








args = tyro.cli(Args)
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic
BATCH_SIZE = 32
EPOCHS = 30

print('Using PyTorch version: ', torch.__version__, 'Device: ', DEVICE)
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = F.relu(x) # sigmoid(x)
        x = self.fc2(x)
        x = F.relu(x) # sigmoid(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x
    
def make_svd(input,output,n_samples,rank):
    vectorize_matrix=torch.randn(input*output,n_samples)
    u, sigma, v = torch.svd(vectorize_matrix)
    matrix=u[:,:rank].view(input, output, -1)
    matrix = torch.transpose(matrix, 0, 2)
    matrix = torch.transpose(matrix, 1, 2)
    return matrix
def make_random(input,output,n_samples,rank):
    vectorize_matrix=torch.randn(input*output,rank)
    # u, sigma, v = torch.svd(vectorize_matrix)
    matrix=vectorize_matrix.view(input, output, -1)
    matrix = torch.transpose(matrix, 0, 2)
    matrix = torch.transpose(matrix, 1, 2)
    normalized_tensor = matrix / torch.norm(matrix, dim=2, keepdim=True)

    return normalized_tensor
def make_random_bias(input,output,n_samples,rank):
    vectorize_matrix=torch.zeros(input*output,rank)
    matrix=vectorize_matrix.view(input, output, -1)
    matrix = torch.transpose(matrix, 0, 2)
    matrix = torch.transpose(matrix, 1, 2)
    normalized_tensor = matrix / torch.norm(matrix, dim=2, keepdim=True)

    return normalized_tensor
def zero_one_random(input,output,n_samples,rank):

    # 모든 요소가 0인 초기 행렬 생성
    matrix = torch.zeros(input*rank, output)

    # 각 행에 대해 1의 위치를 랜덤하게 선택
    indices = torch.randint(0, output, (rank*input,))

    # 선택된 위치에 1 할당
    matrix[torch.arange(input*rank), indices] = 1
    matrix=matrix.view(rank,input,output)
    return matrix


class LCRM(nn.Module):
    
    def __init__(self):

        super(LCRM, self).__init__()
        # self.random=nn.Parameter(torch.randn(28*28,28*28),requires_grad=False)
        self.fixed_weights1 = nn.Parameter(zero_one_random(28*28,args.number_first_hidden,100,1),requires_grad=False)
        self.scale1 = nn.Parameter(torch.randn(1), requires_grad=True)
        self.fixed_weights2 = nn.Parameter(zero_one_random(args.number_first_hidden,args.number_second_hidden,100,1024),requires_grad=False)
        self.scale2 = nn.Parameter(torch.randn(1024), requires_grad=True)
        self.fixed_weights3 = nn.Parameter(zero_one_random(args.number_second_hidden,10,100,512),requires_grad=False)
        self.scale3 = nn.Parameter(torch.randn(512), requires_grad=True)
        # self.fixed_bias1 = nn.Parameter(torch.randn( args.c_bias,args.number_first_hidden), requires_grad=False)
        # self.bias_scale1 = nn.Parameter(torch.randn( args.c_bias), requires_grad=True)
        # self.fixed_bias2 = nn.Parameter(torch.randn( args.c_bias,args.number_second_hidden), requires_grad=False)
        # self.bias_scale2 = nn.Parameter(torch.randn( args.c_bias), requires_grad=True)
        # self.fixed_bias3 = nn.Parameter(torch.randn( args.c_bias,10), requires_grad=False)
        # self.bias_scale3 = nn.Parameter(torch.randn( args.c_bias), requires_grad=True)
        self.bias_scale1 = nn.Parameter(torch.zeros( args.number_first_hidden), requires_grad=True)
        self.bias_scale2 = nn.Parameter(torch.zeros( args.number_second_hidden), requires_grad=True)
        self.bias_scale3 = nn.Parameter(torch.zeros( 10), requires_grad=True)
        init.normal_(self.scale1, mean=0, std=0.1)
        init.normal_(self.scale2, mean=0, std=0.1)
        init.normal_(self.scale3, mean=0, std=0.1)

    def forward(self,  x):
        x = x.view(-1, 28 * 28)
        # x=F.linear(x,self.random)
        x = F.linear(x, (self.fixed_weights1 * self.scale1.unsqueeze(-1).unsqueeze(-1)).sum(dim=0).T,bias=self.bias_scale1)
        x = F.relu(x) # sigmoid(x)
        x = F.linear(x, (self.fixed_weights2 * self.scale2.unsqueeze(-1).unsqueeze(-1)).sum(dim=0).T,bias=self.bias_scale2)
        x = F.relu(x) # sigmoid(x)
        x = F.linear(x, (self.fixed_weights3 * self.scale3.unsqueeze(-1).unsqueeze(-1)).sum(dim=0).T,bias=self.bias_scale3)
        # print(self.scale1,self.scale2,self.scale3)
        x = F.log_softmax(x, dim=1)
        return x
               
train_dataset = datasets.MNIST(
    root="../data/MNIST",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)
test_dataset = datasets.MNIST(
    root="../data/MNIST",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)
def weight_init(m):
    if isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight.data)

model = LCRM().to(DEVICE) 
model.apply(weight_init)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()
def train(model, train_loader, optimizer, log_interval):
    model.train()
    for batch_idx, (image, label) in enumerate(train_loader):
        image = image.to(DEVICE)
        label = label.to(DEVICE)
        optimizer.zero_grad()
        output = model(image)
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()
def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            output = model(image)
            test_loss += loss_fn(output, label).item()
            prediction = output.max(1, keepdim=True)[1]
            correct += prediction.eq(label.view_as(prediction)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy

run_name = f"{args.exp_name}__{args.seed}__{args.c_weight}__{args.c_bias}__{args.number_first_hidden}__{args.number_second_hidden}__{int(time.time())}"
writer = SummaryWriter(f"runs/{run_name}")
writer.add_text(
    "hyperparameters",
    "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),)

for Epoch in range(1, EPOCHS + 1):
    train(model, train_loader, optimizer, log_interval=100)
    test_loss, test_accuracy = evaluate(model, test_loader)
    writer.add_scalar("charts/loss", test_loss, Epoch)
    writer.add_scalar("charts/accuracy", test_accuracy, Epoch)
    print("[EPOCH: {}], \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f} %".format(
        Epoch, test_loss, test_accuracy
    ))