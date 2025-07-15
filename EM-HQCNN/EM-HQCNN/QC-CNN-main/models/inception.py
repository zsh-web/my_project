import torch
import torch.nn as nn
import pennylane as qml
from math import ceil

torch.manual_seed(0)

# 参数设置
n_qubits = 4
n_layers = 1
n_class = 3
n_features = 196
image_x_y_dim = 14
kernel_size = n_qubits  # 量子卷积核大小
stride = 2

# PennyLane 量子设备
dev = qml.device("default.qubit", wires=n_qubits)

# 量子电路
def circuit(inputs, weights):
    var_per_qubit = int(len(inputs) / n_qubits) + 1
    encoding_gates = [qml.RZ, qml.RY] * ceil(var_per_qubit / 2)

    # 输入编码
    for qub in range(n_qubits):
        qml.Hadamard(wires=qub)
        for i in range(var_per_qubit):
            idx = qub * var_per_qubit + i
            if idx < len(inputs):
                encoding_gates[i](inputs[idx], wires=qub)

    # 变分层
    for l in range(n_layers):
        for i in range(n_qubits):
            qml.CRZ(weights[l, i], wires=[i, (i + 1) % n_qubits])
        for j in range(n_qubits, 2 * n_qubits):
            qml.RY(weights[l, j], wires=j % n_qubits)

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# 量子卷积层
class Quanv2d(nn.Module):
    def __init__(self, kernel_size=None, stride=None):
        super(Quanv2d, self).__init__()
        weight_shapes = {"weights": (n_layers, 2 * n_qubits)}
        qnode = qml.QNode(circuit, dev, interface='torch', diff_method='best')
        self.ql1 = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, X):
        assert len(X.shape) == 4  # (batch_size, channels, width, height)
        bs = X.shape[0]
        patches_per_dim = ((X.shape[2] - self.kernel_size) // self.stride) + 1
        patch_outputs = []

        for i in range(0, X.shape[2] - self.kernel_size + 1, self.stride):
            for j in range(0, X.shape[3] - self.kernel_size + 1, self.stride):
                patch = X[:, :, i:i+self.kernel_size, j:j+self.kernel_size]
                patch = torch.flatten(patch, start_dim=1)
                patch_outputs.append(self.ql1(patch))

        # 拼接后reshape为 (bs, n_qubits, patches_per_dim, patches_per_dim)
        X_out = torch.cat(patch_outputs, dim=1).view(bs, n_qubits, patches_per_dim, patches_per_dim)
        return X_out

# 经典+量子混合Inception模块
class Inception(nn.Module):
    def __init__(self, in_channels):
        super(Inception, self).__init__()

        self.branchClassic_1 = nn.Conv2d(in_channels, 4, kernel_size=1, stride=1)
        self.branchClassic_2 = nn.Conv2d(4, 8, kernel_size=4, stride=2)

        self.branchQuantum = Quanv2d(kernel_size=4, stride=2)

    def forward(self, x):
        classic = self.branchClassic_1(x)
        classic = self.branchClassic_2(classic)

        quantum = self.branchQuantum(x)

        # 在channel维度拼接
        outputs = [classic, quantum]
        return torch.cat(outputs, dim=1)

# 整体网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.incep = Inception(in_channels=1)
        self.fc1 = nn.Linear(12 * 6 * 6, 32)
        self.fc2 = nn.Linear(32, n_class)  # 输出类别数3
        self.lr = nn.LeakyReLU(0.1)

    def forward(self, x):
        bs = x.shape[0]
        x = x.view(bs, 1, 14, 14)
        x = self.incep(x)
        x = self.lr(x)

        x = x.view(bs, -1)
        x = self.lr(self.fc1(x))
        x = self.fc2(x)
        return x
