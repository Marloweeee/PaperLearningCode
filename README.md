# 4. TransformerEncoder部分难点与源码实现

## 4.1 关于word embedding，以序列建模为例



### 4.1.1 定义word embedding中的重要参数

```python
# batch_size大小
batch_size=2

# 定义句子序列的最大长度
max_src_len=5
max_tgt_len=5

# 定义序列最大长度
model_dim=16

# 定义单词序列最大索引
max_src_num=8
max_tgt_num=8
```

### 4.1.2 构造输入word embedding的词表序列

```python
import torch
import torch.nn.functional as F

# 构造源句子与目标句子的张量
src_len=torch.Tensor([2,4]).to(torch.int32)# tensor([2, 4], dtype=torch.int32)
tgt_len=torch.Tensor([4,3]).to(torch.int32)# tensor([4, 3], dtype=torch.int32)
```

上述操作构造了源句子和目标句子的张量，源句子包含2个句子，第一个句子长度为2，第二个句子长度为4；目标句子同理。下面将上述源句子与目标句子用维度为句子序列最大长度(5)的张量表示出来，主要使用**pad、unsqueeze、cat操作**，这一步在平时构造训练数据的过程中还是很有实战意义的

```python
# 1.直接构造两个tensor
src_seq=[torch.randint(1,max_src_num,(L,)) for L in src_len]
# [tensor([4, 3]), tensor([7, 3, 1, 1])]

# 2.利用F.pad将序列填充至最大长度5
src_seq=[F.pad(torch.randint(1,max_src_num,(L,)),(0,max_src_len-L) )for L in src_len]
# [tensor([4, 3, 0, 0, 0]), tensor([7, 3, 1, 1, 0])]

# 3.利用torch.unsequeeze将一维张量(5)变为二维张量(1,5)
src_seq=[torch.unsqueeze(F.pad(torch.randint(1,max_src_num,(L,)),(0,max_src_len-L)),0)\
                   for L in src_len]
# [tensor([[6, 5, 0, 0, 0]]), tensor([[3, 7, 1, 4, 0]])]

# 4.使用cat拼接
src_seq=torch.cat([torch.unsqueeze(F.pad(torch.randint(1,max_src_num,(L,)),\
                                         (0,max_src_len-L)),0) for L in src_len])
'''
tensor([[6, 5, 0, 0, 0],
        [3, 7, 1, 4, 0]])
'''

# 同理，使用pad、unsqueeze、cat构造tgt张量
tgt_seq=torch.cat([torch.unsqueeze(F.pad(torch.randint(1,max_tgt_num,(L,))\
                                         (0,max_tgt_len-L)),0) for L in tgt_len])
'''
tensor([[4, 2, 4, 3, 0],
        [4, 5, 2, 0, 0]])
'''
```

**上述操作利用单词索引构造了源句子和目标句子，并且做了padding，填充值默认为0，注意几点：**

> 1.torch.randint(low,high,size)
>
> 用来生成句子序列，生成元素为1-8的张量，其中size的类型必须为tuple，指明了最终的输出序列的尺寸



> 2.torch.nn.functional.pad(*input*, *pad*, *mode='constant'*, *value=None*) → [Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)

**Parameters:**

- **input** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – N-dimensional tensor
- **pad** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple)) – m-elements tuple, where \frac{m}{2} \leq2*m*≤ input dimensions and m*m* is even.
- **mode** – `'constant'`, `'reflect'`, `'replicate'` or `'circular'`. Default: `'constant'`
- **value** – fill value for `'constant'` padding. Default: `0`

F.pad(intput,pad)对input的一个N维tensor在pad上进行填充，pad类型为tuple，tuple中每两个元素为1组，指明填充的维度及填充的数量，例如这里只有两个元素(0,max_src_len-L)，就是说对第一维前面填充0个元素，后面填充(max_src_len-L)个元素，所以填充后只有张量末尾发生了改变，如果这里的参数是(2,max_src_len-L)，那么填充后的数组长度会变成7，每个数组的前两个元素会变成0,0。并且pad填充是从里往外填充的，即先填充dim=-1，再填充dim=-2，如下例子所示

```python
>>> t4d = torch.empty(3, 3, 4, 2)
>>> p1d = (1, 1) # pad last dim by 1 on each side
>>> out = F.pad(t4d, p1d, "constant", 0)  # effectively zero padding
>>> print(out.size())
torch.Size([3, 3, 4, 4])
>>> p2d = (1, 1, 2, 2) # pad last dim by (1, 1) and 2nd to last by (2, 2)
>>> out = F.pad(t4d, p2d, "constant", 0)
>>> print(out.size())
torch.Size([3, 3, 8, 4])
>>> t4d = torch.empty(3, 3, 4, 2)
>>> p3d = (0, 1, 2, 1, 3, 3) # pad by (0, 1), (2, 1), and (3, 3)
>>> out = F.pad(t4d, p3d, "constant", 0)
>>> print(out.size())
torch.Size([3, 9, 7, 3])
```



> 3.torch.unsqueeze(input,dim)返回一个新张量，其尺寸为1并插入到指定位置dim，返回的张量与这个张量共享相同的基础数据。dim的范围介于 `[-input.dim() - 1, input.dim() + 1)`，对于负的dim参数按照以下运算转化 `dim = dim + input.dim() + 1`.例如本用例dim=0与dim=-2填充效果相同。

### 4.1.3 构造embedding

现在已经确定src和tgt都是二维张量，dim=1的位置放的是句子，句子有长有短，但是我们会使用pad操作将其对齐。下面会将句子中的每个词转化为长度为model_dim的词向量

```python
# 前面已经定义model_dim=16，调用nn.Embedding构造词向量
src_embedding_table=nn.Embedding(max_src_num,model_dim)
tgt_embedding_table=nn.Embedding(max_tgt_num,model_dim)
src_embedding=src_embedding_table(src_seq)
tgt_embedding=tgt_embedding_table(tgt_seq)

print(src_embedding_table.weight)# 得到一个table，是src词表的权重
print(src_seq)# src词表的索引
print(src_embedding)# 按照索引取出的权重
print(src_embedding.shape)# shape中前两维不变，最后一维由原来的标量变为一维张量，所以维数变为三维
```

打印输出：

```python
# print(src_embedding_table.weight)
tensor([[-0.8922,  1.6868, -0.6418,  0.3140, -0.8981, -0.2032, -1.1533,  1.4407,
          0.6462, -0.0218, -0.2189, -0.5544, -1.1963, -0.8797, -1.4896, -1.1375],
        [-0.6679,  1.3612,  0.0115, -0.7135, -0.7511, -0.2279,  0.9266,  0.6085,
         -0.0658, -0.7805,  0.1241, -0.5363, -0.0310,  0.1398, -0.2880, -0.3838],
        [-0.6014, -0.5428, -1.9882, -0.7380, -0.8123, -0.5486, -0.7666,  0.4053,
         -0.7813, -0.5849, -0.3628, -0.7975,  0.4671,  2.0936,  0.5843, -1.3917],
        [ 1.8526,  0.5546,  0.1360, -0.6861, -1.5588, -0.8645, -0.5102, -0.4818,
         -0.7090,  1.7046, -0.9654,  0.0745, -0.5227, -0.4729, -0.6181,  0.3763],
        [-0.5552, -0.8068, -1.2071, -1.9199,  1.1797,  0.7980,  0.0243,  0.5780,
         -1.0205,  0.3595,  0.1759, -1.7504,  0.1044,  0.1721,  1.3329,  2.4223],
        [ 0.0843, -0.6042, -0.8001, -1.7500,  1.7444,  0.5514,  0.3341, -0.3628,
          0.0701, -0.1078, -0.0630, -2.8175, -0.3428, -0.7154, -0.1690,  0.9915],
        [ 2.6575, -1.9004, -0.7635,  0.7862,  1.9882, -2.4753, -0.0353,  0.2691,
         -0.1716,  0.0885, -0.1151, -0.6685,  0.5251,  0.4102,  1.5151, -0.1743],
        [-0.4236,  1.5056,  1.6229, -1.1891, -0.3939,  0.0631,  1.0910,  0.4685,
         -0.1328, -0.6178,  0.0780,  1.4527, -0.5974, -0.9052,  0.1527, -0.5200]],
       requires_grad=True)
# print(src_seq)
tensor([[3, 4, 0, 0, 0],
        [2, 6, 7, 5, 0]])
# print(src_embedding)
tensor([[[-0.4144,  0.8002, -0.7077,  1.0783,  0.2542,  0.6446,  0.1157,
           0.3006,  1.3689, -1.8104, -0.4804,  1.5375,  0.2803,  0.3098,
          -0.4550,  0.2727],# 对应weight中的第4个权重向量
         [-1.0427,  2.1106,  0.4897, -1.3543, -1.2303,  0.4397,  0.9002,
          -0.2692,  0.4160,  0.6407, -0.2677, -1.3330,  0.9792, -0.8851,
           0.8809,  2.2589],# 对应weight中的第5个权重向量
         [ 1.4646, -1.2331,  0.7219, -0.1666, -0.0202, -1.0846, -0.6944,
           0.4036,  0.1553,  0.7446, -1.4565,  0.6299, -1.2328, -2.6654,
          -0.7258, -0.7802],
         [ 1.4646, -1.2331,  0.7219, -0.1666, -0.0202, -1.0846, -0.6944,
           0.4036,  0.1553,  0.7446, -1.4565,  0.6299, -1.2328, -2.6654,
          -0.7258, -0.7802],
         [ 1.4646, -1.2331,  0.7219, -0.1666, -0.0202, -1.0846, -0.6944,
           0.4036,  0.1553,  0.7446, -1.4565,  0.6299, -1.2328, -2.6654,
          -0.7258, -0.7802]],

        [[-1.4822, -0.1679, -1.3464,  1.0757,  1.2704, -0.9263,  0.2799,
          -0.9830, -0.6915, -0.5027,  1.4015, -0.8211,  2.2959,  0.5048,
          -0.6541,  1.9831],
         [-1.4967,  0.1112,  0.4351, -1.1601,  0.0701,  1.8887,  1.2096,
          -0.5478, -0.9204,  0.1664,  0.7460,  0.0595,  0.5841, -1.7000,
          -0.1230, -0.2716],
         [ 0.5547, -0.2306,  0.9880,  0.0605, -0.0773,  0.2532, -0.9352,
           1.9237, -0.8470,  0.0512,  0.1840,  2.5656,  0.9901, -0.3262,
          -0.8087,  1.1227],
         [-0.5701,  0.7893, -1.8570,  0.0964, -0.9695,  1.0017, -0.3000,
          -1.8581,  0.6612,  0.1198,  0.4757, -0.1629, -1.7126,  1.1889,
          -1.0428, -0.2390],
         [ 1.4646, -1.2331,  0.7219, -0.1666, -0.0202, -1.0846, -0.6944,
           0.4036,  0.1553,  0.7446, -1.4565,  0.6299, -1.2328, -2.6654,
          -0.7258, -0.7802]]], grad_fn=<EmbeddingBackward>)
# print(src_embedding.shape)
torch.Size([2, 5, 16])
```

上述过程中通过nn.Embedding得到每个词向量的权重，而后直接按照索引将其取出至src_embedding中，如tensor([[3, 4, 0, 0, 0],])对应weight中的第4、5个词向量权重，经过上述操作得到了最终的embedding

### 4.1.4 构造PositionEmbedding

**为什么需要位置编码？**

- 对于任何一门语言，单词在句子中的位置以及排列顺序是非常重要的，它们不仅是一个句子的语法结构的组成部分，更是表达语义的重要概念。一个单词在句子的位置或排列顺序不同，可能整个句子的意思就发生了偏差，例如：

  > I **do not** like the story of the movie, but I **do** like the cast.
  > I **do** like the story of the movie, but I **do not** like the cast.

​		上面两句话所使用的的单词完全一样，但是所表达的句意却截然相反，因此考虑引入词序信息来区别这两句话的意思。

- Transformer模型抛弃了RNN、CNN作为序列学习的基本模型。我们知道，循环神经网络本身就是一种顺序结构，天生就包含了词在序列中的位置信息。当抛弃循环神经网络结构，完全采用Attention取而代之，这些词序信息就会丢失，模型就没有办法知道每个词在句子中的相对和绝对的位置信息。因此，有必要把词序信号加到词向量上帮助模型学习这些信息，位置编码（Positional Encoding）就是用来解决这种问题的方法。

**position embedding的公式如下：**
$$
PE_{(pos,2i)}=\sin\frac {pos}{10000^{2i/d_{model}}}\\
PE_{(pos,2i+1)}=\cos\frac {pos}{10000^{2i/d_{model}}}
$$
其中，$pos$ 是词在词表中出现的位置序号，$i$ 是维度序号。我们可以先生成相同维度的用0填充的张量pe_embedding，再用上述规则进行填充。

```python
# 构造pos和i的matrix
pos_mat=torch.arange(max_src_len).reshape((-1,1))
i_mat=torch.pow(10000,torch.arange(0,model_dim,2).reshape((1,-1))/model_dim)

# 构造position_embedding_table
pe_embedding_table=torch.zeros(max_src_len,model_dim)
pe_embedding_table[:,0::2]=torch.sin(pos_mat/i_mat)
pe_embedding_table[:,1::2]=torch.cos(pos_mat/i_mat)

# 得到position table后再利用nn.embedding得到其权重
pe_embedding=nn.Embedding(max_position_len,model_dim)
pe_embedding.weight=nn.Parameter(pe_embedding_table,requires_grad=False)

# 构造位置张量从position embedding中取向量，形如[0,1,2,...,max(src_len)]
src_pos=torch.cat([torch.unsqueeze(torch.arange(max_src_len),0) for _ in src_len]).to(torch.long)
tgt_pos=torch.cat([torch.unsqueeze(torch.arange(max_tgt_len),0) for _ in tgt_len]).to(torch.long)

# 取出权重向量组成position embedding
src_pe_embedding=pe_embedding(src_pos)
tgt_pe_embedding=pe_embedding(tgt_pos)
print(src_pe_embedding.shape)# torch.Size([2, 5, 16])

# 此时，可以将embedding和position embedding相加得到word embedding
word_embedding=src_pe_embedding+src_embedding
print(word_embedding.shape)# torch.Size([2, 5, 16])
```

可以看到，构造出的position embedding尺寸与词表序列的尺寸相同，所以可以直接将二者sum得到word embedding

```python
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# 考虑source sentence和target sentence，构建序列，序列的字符以其在词表中的索引的形式表示
batch_size=2
max_src_num=8
max_tgt_num=8

# 构造src与tgt尺寸不同的张量，通过后续的padding操作对齐
src_len=torch.Tensor([2,4]).to(torch.int32)
tgt_len=torch.Tensor([4,3]).to(torch.int32)

src_seq=[torch.randint(1,max_src_num,(L,)) for L in src_len]# 单词索引构成的句子
tgt_seq=[torch.randint(1,max_tgt_num,(L,)) for L in tgt_len]
```

## 4.2 通过一个例子演示Softmax中scale的重要性

## 4.2 通过一个例子演示Softmax中scale的重要性

在attention论文中作者使用scale dot-product attention对$QK^{T}$进行缩放，公式如下
$$
Attention(Q,K,V)=softmax(\frac{QK^{T}}{\sqrt d_k})V
$$
其主要目的就是为了将其方差固定在1，防止过大的方差导致的权重不平衡（大的越大，小的越小，像下面例子中所演示的）。

```python
score=torch.randn(5)# tensor([ 0.6052, -0.2023, -1.3294, -0.2546,  1.5445])
prob=F.softmax(score,0)# tensor([0.2187, 0.0976, 0.0316, 0.0926, 0.5595])
```

可以看到，当我们随机生成五个服从正态分布的数值，其softmax后的数值差距并不大

```python
# score的缩放在softmax上并不是线性的，而是大的越大小的越小
alpha1,alpha2=0.1,10
prob1,prob2=F.softmax(score*alpha1,0),F.softmax(score*alpha2,-1)
print(prob1,prob2)
# tensor([0.2100, 0.1937, 0.1730, 0.1927, 0.2306])
# tensor([8.3344e-05, 2.5940e-08, 3.3038e-13, 1.5366e-08, 9.9992e-01])
```

同样，我们也可以用雅克比函数看一下sorce的jacobian matrix(相当于训练过程中的梯度)

```python
def softmax_func(score):
    return F.softmax(score)
jaco_mat1=torch.autograd.functional.jacobian(softmax_func,score*alpha1)
jaco_mat2=torch.autograd.functional.jacobian(softmax_func,score*alpha2)
```

```python
tensor([[ 0.1659, -0.0407, -0.0363, -0.0405, -0.0484],
        [-0.0407,  0.1562, -0.0335, -0.0373, -0.0447],
        [-0.0363, -0.0335,  0.1431, -0.0333, -0.0399],
        [-0.0405, -0.0373, -0.0333,  0.1555, -0.0444],
        [-0.0484, -0.0447, -0.0399, -0.0444,  0.1774]])
tensor([[ 8.3337e-05, -2.1620e-12, -2.7535e-17, -1.2807e-12, -8.3337e-05],
        [-2.1620e-12,  2.5940e-08, -8.5701e-21, -3.9860e-16, -2.5938e-08],
        [-2.7535e-17, -8.5701e-21,  3.3038e-13, -5.0766e-21, -3.3035e-13],
        [-1.2807e-12, -3.9860e-16, -5.0766e-21,  1.5366e-08, -1.5365e-08],
        [-8.3337e-05, -2.5938e-08, -3.3035e-13, -1.5365e-08,  8.3440e-05]])
```

可以看到，当我们将score缩小之后，各元素权重分布较为平衡，梯度也比较容易优化；但是如果将score放大，各元素权重分布就极为失衡，并且发生了梯度消失的现象。所以在进行元素权重标准化对于attention机制还是比较重要的

## 4.3 构造Encoder的seld-attention mask

self-attention mask与Decoder阶段的mask不同，因为我们输入的一句话可能并没有预设的$d_model$这么长，中间是有填充的，这些填充的部分没有必要计算与其他部分的相似度，所以直接将其mask掉节省计算资源
$$
Attention(Q,K,V)=softmax(\frac{QK^{T}}{\sqrt d_k})V
$$
mask的部分是$QK^T$的结果，由于$QK^T$得到的是一个方阵，如果其尺寸为HxH，那么mask矩阵的尺寸也需要是HxH，下面构造一下mask矩阵：

```python
# mask_shape:[batch_size,max_src_len,max_src_len]
valid_encoder_pos=torch.unsqueeze(torch.cat([torch.unsqueeze(F.pad(torch.ones(L),(0,max_src_len-L)),0) for L in src_len]),2)
print(valid_encoder_pos)# [tensor([1., 1., 0., 0., 0.]), tensor([1., 1., 1., 1., 0.])]
```

上述输出说明第一个句子有效位置为前两位，第二个句子有效位置为前四位，那么对于第一个句子的attention矩阵而言，有效位置应该是索引为0和1分别在行、列上的位置，第二个句子同理，下面我们计算一下打印出来看看

```python
valid_encoder_pos_mat=torch.bmm(valid_encoder_pos,valid_encoder_pos.transpose(1,2))
print(valid_encoder_pos_mat.shape)
print(src_len)
print(valid_encoder_pos_mat)
```

打印结果：

```python
torch.Size([2, 5, 5])
tensor([2, 4], dtype=torch.int32)
tensor([[[1., 1., 0., 0., 0.],
         [1., 1., 0., 0., 0.],
         [0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0.]],

        [[1., 1., 1., 1., 0.],
         [1., 1., 1., 1., 0.],
         [1., 1., 1., 1., 0.],
         [1., 1., 1., 1., 0.],
         [0., 0., 0., 0., 0.]]])
```

在自注意力机制下，每个句子与自己本身计算相似度，每个句子都会得到一个相似度矩阵，为了使被填充的位置不参与运算，我们需要使用上述mask矩阵进行mask。上述是有效矩阵，1代表该位置有效，0代表该位置填充，实际中我们使用的是上述矩阵的取反矩阵。

```python
nvalid_encoder_pos_mat=1-valid_encoder_pos_mat
print(invalid_encoder_pos_mat)
```

上述才是mask中用的无效矩阵，此时0代表有效1代表填充，将上述矩阵转为bool型则得到了最终的mask

```python
mask_encoder_self_attention=invalid_encoder_pos_mat.to(torch.bool)
print(mask_encoder_self_attention) 
```

下面就是mask矩阵，True代表此位置需要被mask掉，False表示此位置不能被mask掉（有效位置）

```python
tensor([[[False, False,  True,  True],
         [False, False,  True,  True],
         [ True,  True,  True,  True],
         [ True,  True,  True,  True]],

        [[False, False, False, False],
         [False, False, False, False],
         [False, False, False, False],
         [False, False, False, False]]])
```

将上述mask矩阵填充到我么得儿输入中：

```python
score_demo=torch.randn(batch_size,max(src_len),max(src_len))
masked_score=score_demo.masked_fill(mask_encoder_self_attention,-np.inf)
print(score_demo)
print(masked_score)
print(p)
```

可以看到最终的softmax输出

```python
# score_demo
tensor([[[ 1.9743e+00, -6.5912e-01,  2.4752e+00, -1.1645e+00,  3.0591e-01],
         [-1.5183e+00,  6.0389e-01,  2.6187e-02, -1.9337e+00,  7.4799e-01],
         [-5.0436e-01,  1.9787e+00,  1.0392e+00, -1.1386e-01, -1.1774e+00],
         [ 3.2022e-01,  2.5618e-01, -2.2645e-01,  5.5852e-02, -8.5724e-02],
         [-9.7226e-01, -2.2245e-01,  5.5039e-01,  1.5990e+00,  1.5704e+00]],

        [[-1.1615e+00,  8.0501e-01, -8.2313e-01,  1.2514e-03,  1.8805e+00],
         [-1.0990e+00, -3.3807e-01,  1.3614e+00, -4.5983e-01, -1.7561e-01],
         [ 1.9764e-02, -8.4881e-01,  7.7464e-01, -6.6284e-01,  4.3511e-01],
         [-1.8832e+00, -1.2018e+00, -7.7904e-02, -9.5873e-01,  2.5455e-01],
         [-1.6526e+00, -2.5848e-01, -1.0722e+00,  3.3856e-01,  1.1735e+00]]])
# masked_score
tensor([[[ 1.9743e+00, -6.5912e-01,        -inf,        -inf,        -inf],
         [-1.5183e+00,  6.0389e-01,        -inf,        -inf,        -inf],
         [       -inf,        -inf,        -inf,        -inf,        -inf],
         [       -inf,        -inf,        -inf,        -inf,        -inf],
         [       -inf,        -inf,        -inf,        -inf,        -inf]],

        [[-1.1615e+00,  8.0501e-01, -8.2313e-01,  1.2514e-03,        -inf],
         [-1.0990e+00, -3.3807e-01,  1.3614e+00, -4.5983e-01,        -inf],
         [ 1.9764e-02, -8.4881e-01,  7.7464e-01, -6.6284e-01,        -inf],
         [-1.8832e+00, -1.2018e+00, -7.7904e-02, -9.5873e-01,        -inf],
         [       -inf,        -inf,        -inf,        -inf,        -inf]]])
# p
tensor([[[0.9330, 0.0670, 0.0000, 0.0000, 0.0000],
         [0.1070, 0.8930, 0.0000, 0.0000, 0.0000],
         [   nan,    nan,    nan,    nan,    nan],
         [   nan,    nan,    nan,    nan,    nan],
         [   nan,    nan,    nan,    nan,    nan]],

        [[0.0784, 0.5606, 0.1100, 0.2509, 0.0000],
         [0.0597, 0.1278, 0.6993, 0.1132, 0.0000],
         [0.2468, 0.1035, 0.5250, 0.1247, 0.0000],
         [0.0864, 0.1707, 0.5252, 0.2177, 0.0000],
         [   nan,    nan,    nan,    nan,    nan]]])
```
