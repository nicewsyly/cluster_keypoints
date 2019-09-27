//
// Created by wangsen on 2019/8/19.
//

#ifndef AROKID_BOW_H
#define AROKID_BOW_H

#include "DBow/DBoW3.h"
#include "DBow/DescManip.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>

typedef struct
{

    int itemID;                         ///< id of the template indatabase
    int oWidth;                         ///< the original size of the planar object
    int oHeight;                        ///< the original size of the planar object
    std::string file_name;              ///< the template file name
    std::vector<cv::KeyPoint> keypoints;///< the keypoint in the template to be detected or tracked
    cv::Mat descriptors;                ///< the descriptors for the keypoints
    cv::Mat templ;

}TemplateInfo;

class Bow
{
public:
    Bow()=default;
    virtual ~Bow()=default;
    void init();
    void release();
    void process(const cv::Mat& image,TemplateInfo& tempinfo);
    bool process_patch(const cv::Mat &inputFrame,std::vector<TemplateInfo>& tempinfos,TemplateInfo& queryitem,int& num_match);
private:
    bool in_rect(const cv::KeyPoint& kp,const cv::Rect& rect);
    void split_region(const cv::Rect& rect_to_split,std::vector<cv::Rect>& region_rects,const std::vector<cv::KeyPoint>& keypoints,std::vector<std::vector<cv::KeyPoint> >& region_keypoints);
    std::vector<TemplateInfo> temp_infos;
    DBoW3::WeightingType weight;
    DBoW3::QueryResults query(cv::Mat feature);
    DBoW3::ScoringType score;

    float resizedWidth;
    float resizedHeight;
    std::string DataBasePath;
    std::string KptPath;

    int max_results;
    DBoW3::Vocabulary *m_voc;
    DBoW3::Database *m_db;
    std::string descriptorType;
    cv::Ptr<cv::Feature2D> fdetector;
    //cv::Ptr< cv::DescriptorMatcher> matcher;
    cv::Ptr<cv::BFMatcher> matcher;
    double simiThreshold;
    std::string descriptor_name;

};

#endif //AROKID_BOW_H
