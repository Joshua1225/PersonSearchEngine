# RFSong-7993进行行人检测
重新设计的RFBNet300,模型参数量只有0.99MB，AP达到0.80(相比RFBNet300降了两个点)，速度却可以达到200FPS，并且在多个其他任务表现良好，例如钢筋检测，人手检测等等

文章链接：https://zhuanlan.zhihu.com/p/76491446

## 环境
在python3.6 Ubuntu16.04 pytorch1.1下进行了实验

- 编译NMS

`./make.sh`
- 运行测试代码
`python main.py`