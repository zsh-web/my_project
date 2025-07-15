import pennylane as qml
from math import ceil, pi
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

n_qubits = 4
n_layers = 4
dev = qml.device('default.qubit', wires=n_qubits)

def circuit(inputs, weights):
    # inputs长度必须是2*n_qubits = 8
    for qub in range(n_qubits):
        qml.Hadamard(wires=qub)
        qml.RY(inputs[2*qub], wires=qub)
        qml.RZ(inputs[2*qub+1], wires=qub)

    for layer in range(n_layers):
        for i in range(n_qubits):
            qml.CRZ(weights[layer, i], wires=[i, (i + 1) % n_qubits])
        for j in range(n_qubits, 2*n_qubits):
            qml.RY(weights[layer, j], wires=j % n_qubits)

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


class Quanv2d(nn.Module):
    def __init__(self, kernel_size):
        super(Quanv2d, self).__init__()
        weight_shapes = {"weights": (n_layers, 2*n_qubits)}
        qnode = qml.QNode(circuit, dev, interface='torch', diff_method="best")
        self.ql1 = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.kernel_size = kernel_size
        self.input_len = 2 * n_qubits  # 8

    def forward(self, x):
        # x shape: (bs, c, w, h)
        bs = x.shape[0]
        c = x.shape[1]
        x_lst = []

        for i in range(0, x.shape[2] - self.kernel_size + 1, 2):
            for j in range(0, x.shape[3] - self.kernel_size + 1, 2):
                patch = x[:, :, i:i + self.kernel_size, j:j + self.kernel_size]
                patch_flat = torch.flatten(patch, start_dim=1)  # (bs, c*kernel_size*kernel_size)

                # 自动调整输入长度，扩展或截断到8维
                if patch_flat.shape[1] < self.input_len:
                    # 不够8维，复制扩展
                    repeat_times = ceil(self.input_len / patch_flat.shape[1])
                    patch_flat = patch_flat.repeat(1, repeat_times)
                patch_flat = patch_flat[:, :self.input_len]  # 截断或正好8维

                out = self.ql1(patch_flat)
                x_lst.append(out)

        x = torch.cat(x_lst, dim=1)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.qc = Quanv2d(kernel_size=2)
        # 量子层输出长度 = number_of_patches * n_qubits
        # 输入14x14，每次步长2，patch大小2x2，所以patch数是7x7=49
        # 每个patch输出n_qubits=4个量子测量期望
        # 所以线性层输入是49*4=196
        self.fc1 = nn.Linear(4*7*7, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        bs = x.shape[0]
        x = x.view(bs, 1, 14, 14)
        x = self.qc(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


# 测试用随机输入
if __name__ == "__main__":
    net = Net()
    x = torch.randn(2, 14*14)  # batch=2，展平成1维
    out = net(x)
    print(out.shape)  # 应该是 (2, 10)
