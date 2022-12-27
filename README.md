# VIT
| 包含两个文件：ref_version.py+VIT.py
- ref_version.py是使用pytorch编写，MNIST作为测试集，准确率极限能摸到99%（稳定98%）
- VIT.py是作者在完成课程“机器学习”时一时兴起，使用numpy完成的VIT模型，训练集近似于MNIST
  - 运气好的话能上全局准确率93%
  - 运气不好的话会NAN……
  ![image](https://user-images.githubusercontent.com/79859933/209656161-91d7f692-79bc-4301-a11b-828b6a59bb1d.png)

- 原因可能：
  1. 没有使用GeLu激活函数，在模型中使用的是ReLu（GeLu函数在numpy有点难以实现，会遇上奇怪的问题）
  2. 自注意力层叠加6层，6层梯度爆炸（我暂时没有想到好的处理方法……）
- 可能处理方案：
  1.做自适应的学习率调整
  2.转GeLU
  3.减少自注意力层 
