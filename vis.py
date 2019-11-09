# from visdom import Visdom
# import numpy as np
# import math
# import os.path
# import getpass
# from sys import platform as _platform
# from six.moves import urllib
#
# viz = Visdom(port=1232,env='my_wind')
#
# x,y=0,0
# win1 = viz.line(
#     X=np.array([x]),
#     Y=np.array([y]),
#     opts=dict(title='two_lines'))
#
# for i in range(10):
#     x+=i
#     y+=i
#     viz.line(
#         X=np.array([x]),
#         Y=np.array([y]),
#         win=win1,   # win要保持一致
#         update='append')
#
# buj

# tr_loss=list(range(100))
# print(tr_loss)
# viz.line(Y=np.array(tr_loss), opts=dict(showlegend=True))


# viz.image(
#     np.random.randn(1, 5, 5)
# )
# 多张
# viz.images(
#     np.random.randn(20, 3, 64, 64),
#     opts=dict(title='Random images', caption='How random.')
# )

from visual_loss import Visualizer
from torchnet import meter
#用 torchnet来存放损失函数，如果没有，请安装conda install torchnet
'''
训练前的模型、损失函数设置 
vis = Visualizer(env='my_wind')#为了可视化增加的内容
loss_meter = meter.AverageValueMeter()#为了可视化增加的内容
for epoch in range(10):
    #每个epoch开始前，将存放的loss清除，重新开始记录
    loss_meter.reset()#为了可视化增加的内容
    model.train()
    for ii,(data,label)in enumerate(trainloader):     
        ...
        out=model(input)
        loss=...
        loss_meter.add(loss.data[0])#为了可视化增加的内容
        
    #loss可视化
    #loss_meter.value()[0]返回存放的loss的均值
    vis.plot_many_stack({'train_loss': loss_meter.value()[0]})#为了可视化增加的内容    
'''
#示例
vis = Visualizer(env='my_wind')#为了可视化增加的内容
loss_meter = meter.AverageValueMeter()#为了可视化增加的内容
for epoch in range(10):
    loss_meter.reset()#为了可视化增加的内容
    loss_meter.add(epoch)#假设loss=epoch
    vis.plot_many_stack({'train_loss': loss_meter.value()[0]})#为了可视化增加的内容 
    #如果还想同时显示test loss，如法炮制,并用字典的形式赋值，如下。还可以同时显示train和test accuracy
    #vis.plot_many_stack({'train_loss': loss_meter.value()[0]，'test_loss':test_loss_meter.value()[0]})#为了可视化增加的内容 


