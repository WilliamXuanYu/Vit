import numpy as np
import matplotlib.pyplot as plt
'''
patch大小为 7x7（对于 28x28 图像，共 4 x 4 = 16 个patch）、10 个可能的目标类别（0 到 9）和 1 个颜色通道（灰度）
在网络参数方面，使用了 64 个单元的维度，6 个 Transformer 块的深度，8 个 Transformer 头，MLP 使用 128 维度。
'''

def split_to_patch(data):
    patch = np.array(np.hsplit(data, 16))
    return patch.reshape(50000,16,49)

def LayerNorm(dim):
    (num_layer,__) = dim.shape
    for _ in range(num_layer):
        u = dim[_].mean()
        s = np.power(dim[_] - u,2).mean()
        dim[_] = (dim[_] - u) / np.sqrt(s + 1e-5)
    return dim

class Softmax:
    def __init__(self):
        self.prob = None
        self.label_onehot = None
        self.batch_size = 1 # 标示类型，避免NoneType，getloss中会覆盖
    def forward(self, input):  # 前向传播的计算
        input_max = np.max(input, axis=1, keepdims=True)
        input_exp = np.exp(input - input_max)
        self.prob = input_exp / np.sum(input_exp, axis=1, keepdims=True)
        return self.prob
    def get_loss(self, label):   # 计算损失
        self.batch_size = self.prob.shape[0]
        self.label_onehot = label
        loss = -np.sum(np.log(self.prob+1e-5) * self.label_onehot) / self.batch_size
        return loss
    def backward(self,label_onehot):  # 反向传播的计算
        self.label_onehot = label_onehot
        bottom_diff = (self.prob - self.label_onehot) / self.batch_size
        return bottom_diff

# 全连接线性神经网络，从input_D映射到output_D维度
class LinearLayer:
    def __init__(self, input_D, output_D, bias = True):
        self.data = None
        self.bias = bias
        if not self.bias:
            self._W = np.random.normal(0, 0.1, (input_D, output_D))
            self._grad_W = np.zeros((input_D, output_D))
        else:
            self._W = np.random.normal(0, 0.1, (input_D, output_D))
            self._b = np.random.normal(0, 0.1, (1, output_D))
            self._grad_W = np.zeros((input_D, output_D))
            self._grad_b = np.zeros((1, output_D))

    def forward(self, X):
        assert X.shape[1] == self._W.shape[0]
        self.data = X
        if not self.bias:
            return np.matmul(X, self._W)
        else:
            return np.matmul(X, self._W) + self._b

    def backward(self, grad):
        assert self.data.shape[0] == grad.shape[0]
        assert self.data.shape[1] == self._W.shape[0]
        assert grad.shape[1] == self._W.shape[1]
        if not self.bias:
            self._grad_W = np.matmul(self.data.T, grad)
        else:
            self._grad_W = np.matmul(self.data.T, grad)
            self._grad_b = np.matmul(grad.T, np.ones(self.data.shape[0]))
        return np.matmul(grad, self._W.T)

    def update(self, learn_rate):
        if not self.bias:
            self._W = self._W - self._grad_W * learn_rate
        else:
            self._W = self._W - self._grad_W * learn_rate
            self._b = self._b - self._grad_b * learn_rate

class Pos:
    def __init__(self, input_D):
        self.data = None
        self._b = np.random.normal(0, 0.1, input_D)
        self._grad_b = np.zeros((1, input_D))

    def forward(self, X):
        self.data = X
        return X + self._b

    def backward(self, grad):
        self._grad_b = np.matmul(grad.T, np.ones(self.data.shape[0]))
        return grad # 求导就是其本身

    def update(self, learn_rate):
        self._b = self._b - self._grad_b * learn_rate

class Atten:
    def __init__(self,dim):
        self.Atten = None
        self.v = None
        self.k = None
        self.q = None
        self.data = None
        self.dim = dim
        self.w_q = LinearLayer(dim, dim, bias=False)
        self.w_k = LinearLayer(dim, dim, bias=False)
        self.w_v = LinearLayer(dim, dim, bias=False)
        self.softmax = Softmax()
        self.update_layer_list = [self.w_v, self.w_k, self.w_q]
    def forward(self, data):
        self.data = data
        self.q = self.w_q.forward(data).reshape(1, 64)  # Q矩阵
        self.k = self.w_k.forward(data).reshape(1, 64)  # K矩阵
        self.v = self.w_v.forward(data).reshape(1, 64)  # V矩阵
        self.Atten = self.q.T @ self.k / 8 # Q.T@K
        A = self.softmax.forward(self.Atten) @ self.v.T # V
        return A.reshape(1,64)
    def backward(self,grad):
        v_backward = self.w_v.backward(grad) @ self.Atten  # grad:[1,64]
        k_backward = self.w_k.backward(grad) @ (self.q.T @ self.v)
        q_backward = self.w_q.backward(grad) @ (self.k.T @ self.v)
        return k_backward + q_backward + v_backward
    def update(self,lr):
        for layer in self.update_layer_list:
            layer.update(lr)

class Relu:
    def __init__(self):
        self.data = None
    def forward(self, X):
        self.data = X
        return np.where(X < 0, 0, X)
    def backward(self, grad):
        assert self.data.shape == grad.shape
        x = np.where(self.data < 0, 0, self.data) * grad
        return x

# 6 个 Transformer 块的深度，8 个 Transformer 头
class Encode:
    def __init__(self,dim=64, depth=6):
        self.data = None
        self.dim = dim
        self.depth = depth
        self.lay1 = Atten(64)
        self.lay2 = Atten(64)
        self.lay3 = Atten(64)
        self.lay4 = Atten(64)
        self.lay5 = Atten(64)
        self.lay6 = Atten(64)
        self.update_layer_list = [self.lay6,self.lay5,self.lay4,
                                  self.lay3,self.lay2,self.lay1]

    def forward(self, X):
        self.data = X
        X = LayerNorm(X)
        l1 = LayerNorm(self.lay1.forward(X))
        l2 = LayerNorm(self.lay2.forward(l1))
        l3 = LayerNorm(self.lay3.forward(l2))
        l4 = LayerNorm(self.lay4.forward(l3))
        l5 = LayerNorm(self.lay5.forward(l4))
        l6 = LayerNorm(self.lay6.forward(l5))
        return l6

    def backward(self,grad):
        b6 = self.lay6.backward(grad)
        b5 = self.lay5.backward(b6)
        b4 = self.lay4.backward(b5)
        b3 = self.lay3.backward(b4)
        b2 = self.lay2.backward(b3)
        b1 = self.lay1.backward(b2)
        return b1

    def update(self,lr):
        for layer in self.update_layer_list:
            layer.update(lr)

class MLP:
    def __init__(self, inputs, mlp_num, outputs):
        self.data = None
        self.inputs = inputs
        self.mlp_num = mlp_num
        self.outputs = outputs
        self.lay1 = LinearLayer(inputs,mlp_num)
        self.act1 = Relu()
        self.lay2 = LinearLayer(mlp_num,outputs)
        self.softmax = Softmax()
        self.update_layer_list = [self.lay2, self.lay1]
    def forward(self, X):
        self.data = X.reshape(1,64)
        l1 = self.lay1.forward(self.data)
        l2 = self.act1.forward(l1)
        l3 = self.lay2.forward(l2)
        l4 = self.softmax.forward(l3)
        return l4
    def backward(self,grad):
        b4 = self.softmax.backward(grad)
        b3 = self.lay2.backward(b4)
        b2 = self.act1.backward(b3)
        b1 = self.lay1.backward(b2)
        return b1
    def update(self,lr):
        for layer in self.update_layer_list:
            layer.update(lr)

class Classification: # 64->10->softmax->result
    def __init__(self, inputs, outputs):
        self.data = None
        self.inputs = inputs
        self.outputs = outputs
        self.lay = LinearLayer(inputs,outputs)
        self.cls = Softmax()

    def forward(self, X):
        self.data = X.reshape(1,64)
        l1 = self.lay.forward(self.data)
        l2 = self.cls.forward(l1)
        return l2

    def get_loss(self, label):
        return self.cls.get_loss(label)

    def backward(self, grad):
        b2 = self.cls.backward(grad)
        b1 = self.lay.backward(b2)
        return b1

    def update(self, lr):
        self.lay.update(lr)

class Vit:
    def __init__(self, *, image_size=28, patch_size=7, num_classes=10, channels=1,
            dim=64, depth=6, heads=8, mlp_dim=128):
        self.data = None
        self.image_size = image_size
        self.patch_size = patch_size**2
        self.num_classes = num_classes
        self.channels = channels
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        # 建立网络结构
        self.linearlay = LinearLayer(self.patch_size,self.dim)
        self.act1 = Relu()
        self.poslay = Pos(self.dim)
        self.encode = [Encode(64, 6) for _ in range(8)]
        self.mlp = MLP(64,128,64)
        self.cls = Classification(64,10)
        self.update_layer_list = [self.cls, self.mlp,
                                  self.encode[0], self.encode[1], self.encode[2], self.encode[3],
                                  self.encode[4], self.encode[5], self.encode[6], self.encode[7],
                                  self.poslay, self.linearlay]
    def forward(self, X): # 这里的forward不完全，最后出来需要一个平均池化，故放在train函数中
        self.data = X
        lay1 = self.linearlay.forward(X)
        lay2 = self.act1.forward(lay1)
        lay3 = self.poslay.forward(lay2)
        lays = np.zeros([8,64])
        for i in range(8):
            lays[i] = self.encode[i].forward(lay3)
        lay4 = np.mean(lays,axis=0).reshape(64)
        lay5 = self.mlp.forward(lay4)
        return lay5
    def backward(self,grad):
        b6 = self.cls.backward(grad) # b7:[1,64]
        bs = np.zeros([8,64])
        for i in range(8): # 这里的8个是并行计算后平均，没有依赖关系，统一使用b7
            bs[i] = self.encode[i].backward(b6)
        b5 = np.mean(bs,axis=0).reshape(1,64)
        b4 = self.mlp.backward(b5)
        b3 = self.poslay.backward(b4)
        b2 = self.act1.backward(b3)
        b1 = self.linearlay.backward(b2)
        return b1
    def update(self, lr):
        for layer in self.update_layer_list:
            layer.update(lr)
    def train(self,data,label,lr):
        print('Start training:')
        batch = 100
        loss = [0]*batch
        # epoch = 50000/batch
        for _ in range(batch): # 测试：前100张图片
            temp = np.zeros([16, 64])  # 用于存储16个patch的独立编码
            batch_labels = label[0]
            for i in range(16):
                batch_images = data[0][i].reshape(1,49)
                temp[i] = self.forward(batch_images) # 输出每一个patch的embedding
            cls_token = np.mean(temp,axis=0) # 一个64维的token用于最后的分类任务，这里用的平均池化
            self.cls.forward(cls_token) # 使用上述token完成分类任务，self.cls中保存的是这个平均的token，补全最后一个forward
            loss[_] = self.cls.get_loss(batch_labels)
            print("loss:",loss[_])
            self.backward(batch_labels)
            self.update(lr)
        plt.plot(range(len(loss)), loss)



if __name__ == '__main__':
    data_ = np.load("train_data.npy")  # 784个特征值,28*28图像
    label = np.load("train_label.npy")  # 0~9的标签
    label = np.eye(10)[label].astype('int') # 转化为one-hot
    data = split_to_patch(data_) # 拆分为16个patch,每个patch 7*7 个特征
    model = Vit(image_size=28, patch_size=7, num_classes=10, channels=1,
            dim=64, depth=6, heads=8, mlp_dim=128)
    model.train(data,label,0.1)
    plt.show()
