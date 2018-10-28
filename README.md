# Image-Classification-using-Flowers-dataset Competition
In “Elephant Fractial Competition”

这是本人作为小白第一次参加类似的竞赛，总结的这近一个月的时间内踩过的坑，走过的弯路和积累的一些有用的Tricks，分享给和我一样的新手小白们。
【所有参考的Blogs, Tutorials等都给出了明确的出处，有部分资料需要翻墙访问】  


## TFRecords Files
在此次竞赛中，训练的图像数据集相对较小，共3500(350x10)张灰度图像(Grayscale)，包含5类花朵图片(1-玫瑰(rose), 2-向日葵(sunflower), 
3-雏菊(daisy), 4-蒲公英(dandelion), 5-郁金香(tulip))，以*.tfrecord(s)类型文件存储。我是第一次接触这种类型的数据格式，所以也花了很长
时间去理解学习，简单地说：*tfrecord file: images which are numpy arrays and labels which are a list of strings*，是一种将*.jpeg 
或 *.png 等类型图像解码后再储存成tfrecord文件，所以体积会比较大(比如：我在Google Image上面额外找的794M用于扩充训练集的5类花朵图片，
转成.tfrecords文件有3.36G大小 Horrible....)。  
这里推荐两篇Blogs供大家参考：  
[1. Why Every Tensorflow Developer Should Know About Tfrecord](https://www.skcript.com/svr/why-every-tensorflow-developer-should-know-about-tfrecord/)  
[2. Tensorflow Records? What they are and how to use them.](https://medium.com/mostly-ai/tensorflow-records-what-they-are-and-how-to-use-them-c46bc4bbb564)  

## 
