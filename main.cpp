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
