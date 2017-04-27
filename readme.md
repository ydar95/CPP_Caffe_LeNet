# C++ 调用 Caffe 训练好的网络(样例:LeNet)

## 1. 说明

我们用Caffe可以很方便的训练我们的网络,但是当真正的拿去用到C++之类的工程的时候就不知道怎么做了.于是我打算介绍下C++ 如何调用 Caffe 训练好的网络和API .

前期准备: g++ ( 支持 C++11 版本 )  cmake opencv ,还有Caffe ( 我在这里就不介绍Caffe 的安装使用了)

## 2. 一些 caffe API 的介绍
官方文档: http://caffe.berkeleyvision.org/doxygen/index.html
### 2.1 caffe::net<Ty> 

这个类是我们调用Caffe API 的核心.它提供很方便的使用方式,只要我们提供正确 caffemodel 和 net(网络结构<prototxt>) ,就可以自动帮我们生成好用于预测的网络.

### 2.2  caffe::Caffe::set_mode();

这个是用于设置,我们使用CPU进行运算还是GPU进行运行.默认情况下caffe是使用CPU运行的.
使用方式如下

```c++
using namespace caffe;
Caffe::set_mode(Caffe::CPU);	// 设置为CPU模式
Caffe::set_mode(Caffe::GPU);	// 设置为GPU模式
```

### 2.3 caffe::Blob<Ty>
这个是caffe 放置 网络模型运行时所有数据的class.我们的网络的输入和输出也是使用这个class管理.

## 3. 代码

main.cpp

```c++
#include <iostream>
#include <opencv2/opencv.hpp>
#include <caffe/caffe.hpp>
#include <string>
using namespace caffe;
using namespace std;
int main(int argc,char* argv[]) {
    typedef float type;
    type ary[28*28];

	//在28*28的图片颜色为RGB(255,255,255)背景上写RGB(0,0,0)数字.
    cv::Mat gray(28,28,CV_8UC1,cv::Scalar(255));
    cv::putText(gray,argv[3],cv::Point(4,22),5,1.4,cv::Scalar(0),2);

	//将图像的数值从uchar[0,255]转换成float[0.0f,1.0f],的数, 且颜色取相反的 .
    for(int i=0;i<28*28;i++){
			// f_val =(255-uchar_val)/255.0f
            ary[i] = static_cast<type>(gray.data[i]^0xFF)*0.00390625;	
    }

    cv::imshow("x",gray);
    cv::waitKey();

    //set cpu running software
    Caffe::set_mode(Caffe::CPU);

    //load net file	, caffe::TEST 用于测试时使用
    Net<type> lenet(argv[1],caffe::TEST);

    //load net train file caffemodel
    lenet.CopyTrainedLayersFrom(argv[2]);



    Blob<type> *input_ptr = lenet.input_blobs()[0];
    input_ptr->Reshape(1,1,28,28);

    Blob<type> *output_ptr= lenet.output_blobs()[0];
    output_ptr->Reshape(1,10,1,1);

	//copy data from <ary> to <input_ptr>
    input_ptr->set_cpu_data(ary);

	//begin once predict
    lenet.Forward();


    const type* begin = output_ptr->cpu_data();
    
	// get the maximum index
    int index=0;
    for(int i=1;i<10;i++){
    	if(begin[index]<begin[i])
		index=i;
    }

	// 打印这次预测[0,9]的每一个置信度
    for(int i=0;i<10;i++)
		cout<<i<<"\t"<<begin[i]<<endl;

	// 展示最后的预测结果
    cout<<"res:\t"<<index<<"\t"<<begin[index]<<endl;
    return 0;
}
```

CMakeLists.txt

```c++
cmake_minimum_required(VERSION 3.1)
project(caffe_mint)

set(CMAKE_CXX_STANDARD 11)

include_directories("yourcaffe/include") 
link_directories("yourcaffe/lib")	# libcaffe.so

#如果你使用 cuda
include_directories("/usr/local/cuda/include")
link_directories("/usr/local/cuda/lib64") 

set(SOURCE_FILES main.cpp)
add_executable(caffe_mint ${SOURCE_FILES})

#使用opencv
find_package(OpenCV REQUIRED)
target_link_libraries(caffe_mint ${OpenCV_LIBS} )

#其他的库 cublas 是cuda的
target_link_libraries(caffe_mint caffe cublas boost_system glog)
```

## 4. 编译运行
### 4.1 编译

`cmake .  && make`

### 4.2 运行
`./caffe_mint  lenet.prototxt  lenet_iter_10000.caffemodel  1 `
最后一个传入参数1 是我们要opencv 绘制的数字. 如是"2" 就在图像上绘制 2

### 4.3 完整代码
Github 仓库 https://github.com/ydar95/CPP_Caffe_LeNet.git ,可能在CMakelists.txt 和 文章有些出入.


