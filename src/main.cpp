#include <iostream>
#include <cv.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs/imgcodecs_c.h>
#include <opencv2/core/core_c.h>
#include "affine_image_generator06.hpp"
#include "utils.hpp"
#include "Bow.h"

int main()
{
    std::string img_folder="../movie100/"; 
    std::vector<std::string> imgs;
    list_dir(img_folder.c_str(),imgs);

    std::string bgimg_folder="../movie100-bgimgs/";
    std::vector<std::string> bgimgs;
    list_dir(bgimg_folder.c_str(),bgimgs);
    
    affine_image_generator06 * image_generator = new affine_image_generator06();
    affine_transformation_range range; 
    image_generator->set_transformation_range(&range);
    int num_generate_images=10;     
    srand((unsigned)time(NULL));     
    Bow *bow=new Bow();
    bow->init();
    cv::Mat miximg;
    for(int ii=0;ii<imgs.size();++ii)
    {
        std::cout<<"img name "<<imgs[ii]<<std::endl;
        std::string imgpath=img_folder+imgs[ii];
        cv::Mat ori_img=cv::imread(imgpath);
        ori_img+=cv::Scalar::all(1);
              
        IplImage *iplimg=new IplImage(ori_img);;
        image_generator->set_original_image(iplimg);
        
        for(int ni=0;ni<num_generate_images;++ni)
        {
            int bgi=rand()%bgimgs.size();
            std::cout<<"bgimgs "<<bgimgs[bgi]<<std::endl;
            cv::Mat bgimg=cv::imread(bgimg_folder+bgimgs[bgi]);            
            
	    TemplateInfo tempinfo;
	    std::vector<TemplateInfo> infos;
            image_generator->generate_random_affine_image();
            std::cout<<"affine matrix ------------------------"<<image_generator->a[0]<<std::endl;
            //cv::Mat affine_matrix(cv::Size(3,2),CV_32FC1,image_generator->a);
            cv::Mat foreimg=cv::cvarrToMat(image_generator->generated_image);
            /*
            //test whether in box
            cv::Mat homo;
            cv::Mat botm=(cv::Mat_<float>(1,3)<<0,0,1);
            cv::vconcat(affine_matrix,botm,homo);
            std::vector<cv::Point2f> srcpts;
            srcpts.push_back(cv::Point2f(0,0));
            srcpts.push_back(cv::Point2f(ori_img.cols-1,0));
            srcpts.push_back(cv::Point2f(ori_img.cols-1,ori_img.rows-1));
            srcpts.push_back(cv::Point2f(0,ori_img.rows-1));
            std::vector<cv::Point2f> dstpts;
            cv::perspectiveTransform(srcpts,dstpts,homo); 
            std::vector<cv::Point> drawpts(dstpts.begin(),dstpts.end());
            cv::polylines(foreimg,drawpts,true,cv::Scalar(255,0,0));
            cv::imshow("foreimg",foreimg);
            cv::waitKey();
            //--test whether in box            
            */
            cv::resize(bgimg,bgimg,cv::Size(image_generator->generated_image->width,image_generator->generated_image->height)); 
            
            add_bg(bgimg,foreimg,miximg);
            //cv::imshow("bgimg",bgimg);
            //cv::imshow("foreimg",foreimg);
            int num_match;
            bow->process_patch(miximg,infos,tempinfo,num_match);
            std::cout<<tempinfo.file_name<<std::endl;
            
            //cv::imshow("generate random affine image",miximg);
            cv::waitKey(0);
        }
    }

    return 0;
}
