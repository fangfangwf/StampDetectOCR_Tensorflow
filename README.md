介绍:

	基于深度学习的***银行票据识别需求解决方案

检测模型：
	
	Yolov3， 是调用opencv的dnn模块来实现
识别模型：

	CRNN，是调用Tensorflow的C++ API来实现
	此次用到的训练CRNN模型的tensorflow代码为https://github.com/MaybeShewill-CV/CRNN_Tensorflow，
	另外转换代码，将ckpt转换成pb文件的代码参考https://github.com/leimao/Frozen_Graph_TensorFlow/blob/master/TensorFlow_v1/test_pb.py
	
	
检测的目标是***银行的票据。
检测的目标共有4类，分别为主键、流水号、附件及附件标题。

识别：

	对检测到的目标进行识别，其中需要识别的目标有主键、流水号和附件标题。
	此识别是对水平从左到有的输入有效。


cmake进行构建。

	对于自己写的.cpp .h文件拷贝到src文件夹下，然后在CmakeLists.txt文件中的
	add_executable (JZBankSolution_CPU src/demo.cpp src/dataStructures.h src/***.cpp src/***.h) 后面依次添加，
	然后cmake构建就可以


文件夹说明：

3rdparty:
	  
	存放的是第三方库 opencv4.1.0的库和tensorflow1.10.0 CPU版的库，下载
	链接：https://pan.baidu.com/s/1TIXq3RsR2EnqSAezMxwoRA 
        提取码：buxc
	补充说明：windows下 tensorflow1.10.0 CPU版的库，不是我本人编译的，我尝试了很多次都失败了，
	最后是从网上找到别的网友编译好的，在此向那位网友表示感谢！（遗憾，那个原始下载链接找不到了）
build：
	
	是已经cmake构建vsproject时的目录

model：

	yolo： 存放的是yolov3的cfg文件和weights文件，以及names文件，下载
	       链接：https://pan.baidu.com/s/1IiwEO4_1wV-KtfAWi9sS9Q 
               提取码：sq9l
	        
	crnn： 存放的是crnn的模型文件及字典文件，下载
	       链接：https://pan.baidu.com/s/1_iNHZMssHAZeMFVDF4ziZg 
               提取码：wgv7
	
	
images： 
	
	存放的是几张测试图片，请从https://github.com/BigPandaCPU/StampDetectOCR/tree/master/images下载
	
src:   

	存放的是源文件


注意：
	
	在下载此文件后，需要在本地重新cmake构建一下
	参见cmake截图.png
	

版本说明：

	opencv4.1.0 release 64
	Tensorflow1.10.0 release 64  CPU版
	Cmake 3.13.0

Author:BigPanda

E-mail:wangxiong@founder.com

State Key Laboratory of Digital Publishing Technology

Date:2020-9-10 

