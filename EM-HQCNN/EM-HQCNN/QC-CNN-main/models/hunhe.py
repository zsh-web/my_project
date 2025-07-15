import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
from math import ceil

# 固定随机种子，方便复现
torch.manual_seed(0)

# 量子电路参数
n_qubits = 4
n_layers = 1
n_class = 3
n_features = 196  # 输入特征长度（对应量子编码长度）
image_x_y_dim = 14
kernel_size = n_qubits
stride = 2

# 量子设备
dev = qml.device("default.qubit", wires=n_qubits)

# 量子电路定义
def circuit(inputs, weights):
    var_per_qubit = int(len(inputs) / n_qubits) + 1
    encoding_gates_list = ['RZ', 'RY'] * ceil(var_per_qubit / 2)
    gate_map = {'RZ': qml.RZ, 'RY': qml.RY}

    for qub in range(n_qubits):
        qml.Hadamard(wires=qub)
        for i in range(var_per_qubit):
            idx = qub * var_per_qubit + i
            if idx < len(inputs):
                gate_map[encoding_gates_list[i]](inputs[idx], wires=qub)

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
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, X):
        assert len(X.shape) == 4  # (batch, channel, height, width)
        bs = X.shape[0]
        out_dim_x = (X.shape[2] - self.kernel_size) // self.stride + 1
        out_dim_y = (X.shape[3] - self.kernel_size) // self.stride + 1

        patches = []
        for i in range(0, X.shape[2] - self.kernel_size + 1, self.stride):
            for j in range(0, X.shape[3] - self.kernel_size + 1, self.stride):
                patch = X[:, :, i:i + self.kernel_size, j:j + self.kernel_size]
                patch_flat = torch.flatten(patch, start_dim=1)  # 展平成(batch_size, kernel_size*kernel_size*channel)
                patches.append(self.qlayer(patch_flat))
        # 拼接所有patch的量子输出，shape (batch_size, n_qubits * num_patches)
        X_out = torch.cat(patches, dim=1)
        X_out = X_out.view(bs, n_qubits, out_dim_x, out_dim_y)
        return X_out


# 整体模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.qlayer = Quanv2d(kernel_size=kernel_size, stride=stride)
        self.conv1 = nn.Conv2d(n_qubits, 16, kernel_size=3, stride=1)
        self.leaky_relu = nn.LeakyReLU(0.1)
        # 计算全连接输入维度
        conv_out_dim = ((image_x_y_dim - kernel_size) // stride + 1)  # 量子卷积输出大小
        conv_out_dim = conv_out_dim - 3 + 1  # 经典卷积输出大小 (kernel=3, stride=1)
        self.fc1 = nn.Linear(16 * conv_out_dim * conv_out_dim, n_class * 2)
        self.fc2 = nn.Linear(n_class * 2, n_class)

    def forward(self, X):
        bs = X.shape[0]
        X = X.view(bs, 1, image_x_y_dim, image_x_y_dim)  # 假设输入是一维，转换成图像格式
        X = self.qlayer(X)
        X = self.leaky_relu(self.conv1(X))
        X = X.view(bs, -1)
        X = self.leaky_relu(self.fc1(X))
        X = self.fc2(X)
        return X


# 简单训练示例（伪代码）
def train_example():
    model = Net()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 假数据举例 (batch_size, feature_len)
    batch_size = 5
    dummy_input = torch.randn(batch_size, n_features)
    dummy_labels = torch.randint(0, n_class, (batch_size,))

    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(dummy_input)
        loss = criterion(outputs, dummy_labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch} loss: {loss.item():.4f}")


if __name__ == "__main__":
    train_example()
