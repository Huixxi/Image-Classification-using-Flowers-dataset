# Image-Classification-using-Flowers-dataset Competition
In “Panorama AI Competition”

这是本人作为小白第一次参加类似的竞赛，总结的这近一个月的时间内踩过的坑，走过的弯路和积累的一些有用的Tricks，分享给和我一样的新手小白们。
【所有参考的Blogs, Tutorials等都给出了明确的出处，有部分资料需要翻墙访问】  


## What is TFRecords File
在此次竞赛中，训练的图像数据集相对较小，共3500(350x10)张灰度图像(Grayscale)，包含5类花朵图片(1-玫瑰(rose), 2-向日葵(sunflower), 
3-雏菊(daisy), 4-蒲公英(dandelion), 5-郁金香(tulip))，以*.tfrecord(s)类型文件存储。我是第一次接触这种类型的数据格式，所以也花了很长
时间去理解学习，简单地说：*tfrecord file: images which are numpy arrays and labels which are a list of strings*，是一种将\*.jpeg 
或 \*.png 等类型图像解码后再储存成tfrecord文件，所以体积会比较大(比如：我在Google Image上面额外找的794M用于扩充训练集的5类花朵图片，
转成.tfrecords文件有3.36G大小 Horrible....)。  
这里推荐两篇Blogs供大家参考：  
[1. Why Every Tensorflow Developer Should Know About Tfrecord](https://www.skcript.com/svr/why-every-tensorflow-developer-should-know-about-tfrecord/)  
[2. Tensorflow Records? What they are and how to use them.](https://medium.com/mostly-ai/tensorflow-records-what-they-are-and-how-to-use-them-c46bc4bbb564)  

## Read from a TFRecords file in Tensorflow
拿到tfrecords类型的数据集之后，第一件事就是先把图片从里面读出来，看看图片是长什么样的，因为是很菜的小白，所以也为此头疼了好久，借鉴了一些非常优秀的教程，  
这里是我本人的Notebook代码：  
[Source Code: Read from TFRecords files and Plot](https://github.com/Huixxi/Image-Classification-using-Flowers-dataset/blob/master/bin/read_from_tfrecords_files_and_plot.ipynb)  
参考的教程链接：  
[Youtube Video Tutorial:Tensorflow tutorial_TFRecord tutorial_02](https://www.youtube.com/watch?v=jbLi8JHgl28&list=LLUMZo4j7Z8dYMlWWpASiyIA&t=11s&index=21)    
[上述视频教程中代码的GitHub地址](https://github.com/kevin28520/My-TensorFlow-tutorials/blob/master/03%20TFRecord/notMNIST_input.py)   

## Write into a TFRecords file
本人在进行数据集扩充进行数据增强的部分时，需要将自己在网上爬下来的图片转换成tfrecords文件，以便读入自己的模型用于模型的预训练，就顺便学习了一下该如何制作tfrecords文件，本该是后面的内容，就放在tfrecords文件这一块讲了。这里以一个例子进行说明：[How to create a deep learning dataset using Google Images](https://www.pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/)，同时这也是一篇非常棒的tutorial。  
首先按照教程的操作，也可以直接看我本人整理好的[Notebook代码](https://github.com/Huixxi/Image-Classification-using-Flowers-dataset/blob/master/bin/create_a_deep-learning_dataset_using_google-images.ipynb)，将Google Image上的相关图片Download到本地。  
接下来准备将下载好的全部图片进行标记并转化成tfrecords文件：  
本人的Notebook代码：  
[Source Code: Write_into_TFRecords_files](https://github.com/Huixxi/Image-Classification-using-Flowers-dataset/blob/master/bin/write_into_tfrecords_files.ipynb)   
参考的教程链接(Vely Good!)：  
[How to write into and read from a TFRecords file in TensorFlow](http://www.machinelearninguru.com/deep_learning/tensorflow/basics/tfrecord/tfrecord.html)    

## Neural network architecture and Models
终于到神经网络模型的部分了。因为做计算机视觉图像相关，而且又是图像分类这种很基础的方向，卷积神经网络-CNN一定是首选；对于如此小的数据集来说，要想达到一个不错的分类准确率，自然也避不开使用Pre-Trained Models和Transfer Learning。从2012年以来涌现的那么多神经网络模型，如Xception, VGG16, VGG19, ResNet50, InceptionV3, InceptionResNetV2, MobileNet等，我们该如何进行选择。这篇[Tutorial：Using Keras Pre-trained Deep Learning models for your own dataset](https://gogul09.github.io/software/flower-recognition-deep-learning)给了我很大的启发。最终我在本次竞赛中使用的是[MobileNets](https://arxiv.org/pdf/1704.04861.pdf)，因为和其他模型比起来，它的参数真的是太少了，并且也十分强大。  
对于如何使用Pre_trained Models，我参考了如下教程(Still Vely Good!):  
[1.MobileNet image Classification with Keras](https://www.youtube.com/watch?v=OO4HD-1wRN8&list=PLZbbT5o_s2xrwRnXk_yCPtnqqo4_u2YGL&index=31)  
[2.Build image classifier using transfer learning - Fine-tuning MobileNet with Keras 1](https://www.youtube.com/watch?v=4Tcqw5oIfIg&list=PLZbbT5o_s2xrwRnXk_yCPtnqqo4_u2YGL&index=32)  
[3.Train image classifier using transfer learning - Fine-tuning MobileNet with Keras 2](https://www.youtube.com/watch?v=-0Blng0Ww8c&list=PLZbbT5o_s2xrwRnXk_yCPtnqqo4_u2YGL&index=33)  
[4.Sign language image classification - Fine-tuning MobileNet with Keras 3](https://www.youtube.com/watch?v=FNqp4ZY0wDY&list=PLZbbT5o_s2xrwRnXk_yCPtnqqo4_u2YGL&index=34)  

由于这些预训练模型通常都是用RGB 3-channel 图片进行训练的，为此我对Grayscale训练数据集做了一些预处理工作，包括图片剪裁并将其转成3-channel的“Grayscale”图片（the trick: convert 1-channel grayscale to 3-channel grayscale is included in the final part）

## Train and Test
模型一开始是用CPU训练的，产能问题一度使我想要放弃，最后还是迁到了自己电脑上的GTX 950M，真香！  
最终的模型：[final_model.h5](https://github.com/Huixxi/Image-Classification-using-Flowers-dataset/tree/master/bin/models)也是基于预训练模型的框架：[flowers_5.h5](https://github.com/Huixxi/Image-Classification-using-Flowers-dataset/tree/master/bin/models)，对全部参数进行重新训练，所以也就没有用到太多迁移学习的地方，但是用自己找的图片数据进行了预训练[pre_train_model.h5](https://github.com/Huixxi/Image-Classification-using-Flowers-dataset/tree/master/bin/models)。    

[Train Code](https://github.com/Huixxi/Image-Classification-using-Flowers-dataset/blob/master/bin/train.py)  
[Test Code](https://github.com/Huixxi/Image-Classification-using-Flowers-dataset/blob/master/bin/test.py)  

参考Blogs:  
[1.Data Augmentation Techniques in CNN using Tensorflow](https://medium.com/ymedialabs-innovation/data-augmentation-techniques-in-cnn-using-tensorflow-371ae43d5be9)   

## Other Tricks(Including how to denoise grayscale image)
其他的一些Tricks整理在这个[NoteBook文件](https://github.com/Huixxi/Image-Classification-using-Flowers-dataset/blob/master/bin/utils_in_pretrained_model.ipynb)  
