# Image-Classification-using-Flowers-dataset Competition
In “Elephant Fractial Competition”

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
这里是本人参考的教程链接：  
[Youtube Video Tutorial:Tensorflow tutorial_TFRecord tutorial_02](https://www.youtube.com/watch?v=jbLi8JHgl28&list=LLUMZo4j7Z8dYMlWWpASiyIA&t=11s&index=21)    
[上述视频教程中代码的GitHub地址](https://github.com/kevin28520/My-TensorFlow-tutorials/blob/master/03%20TFRecord/notMNIST_input.py)   

## Write into a TFRecords file
本人在进行数据集扩充进行数据增强的部分时，需要将自己在网上爬下来的图片转换成tfrecords文件，以便读入自己的模型用于模型的预训练，就顺便学习了一下该如何制作tfrecords文件，这里以一个例子进行说明：[How to create a deep learning dataset using Google Images](https://www.pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/)，同时这也是一篇非常棒的tutorial。  
首先按照教程的操作，也可以直接看我本人整理好的[Notebook代码](https://github.com/Huixxi/Image-Classification-using-Flowers-dataset/blob/master/bin/create_a_deep-learning_dataset_using_google-images.ipynb)，将Google Image上的相关图片Download到本地。




## Neural network architecture and Models



最后希望大家能在此次竞赛中，学的开心，玩的开心，踩坑踩得也开心(给你个生无可恋的眼神儿自己体会)。
