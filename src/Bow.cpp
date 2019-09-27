//
// Created by wangsen on 2019/8/19.
//

#include "Bow.h"
//#include "common_log.h"

void Bow::init()
{
    //LOGD("Detector::initBow Enter");
    std::string modelPath="../bow_models/";
    max_results = 5;
    weight = DBoW3::TF_IDF;
    score = DBoW3::L1_NORM;
    //setup paths string

    this->DataBasePath = modelPath + "briskdb.yml.gz";
    this->KptPath = modelPath + "briskkpt.yml.gz";

    this->m_db = new DBoW3::Database(this->DataBasePath);
    //LOGD("Detector::initBow Database size: %d", this->m_db->size());
    this->simiThreshold = 0.3;
    //std::string descriptor_name;
    int detectionHeight, detectionWidth;
    //LOGD("Detector::initBow load kpt files: %s", this->KptPath.c_str());
    //LOGD("Detector::initBow load kpt files: %s", this->KptPath.c_str());

    cv::FileStorage fs;

    fs.open(this->KptPath.c_str(), cv::FileStorage::READ);
    //LOGD("Detector::initBow filestorage isopened : %d",fs.isOpened());
    fs["detectionHeight"] >> detectionHeight;
    fs["detectionWidth"] >> detectionWidth;
    fs["descriptor_name"] >> this->descriptor_name;
    //LOGD("Detector::initBow model path : %s",this->KptPath.c_str());
    this->resizedHeight = detectionHeight;
    this->resizedWidth = detectionWidth;

    //this->resizedHeight=360;
    //this->resizedWidth=640;
    //LOGD("resize width height %d %d",detectionWidth,detectionHeight);
    if(descriptor_name == "orb")
    {
        this->fdetector = cv::ORB::create(1000);
        //this->matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
        this->matcher=new cv::BFMatcher(cv::NORM_L2);
    }
    else if(descriptor_name=="brisk")
    {
        this->fdetector=cv::BRISK::create();
        this->matcher=new cv::BFMatcher(cv::NORM_L2);
        //this->matcher=cv::DescriptorMatcher::create("BruteForce-Hamming");
    }
    else if (descriptor_name=="akaze")
    {
        this->fdetector=cv::AKAZE::create();
        //this->matcher=cv::DescriptorMatcher::create("BruteForce-Hamming");
    }
    else
    {
        //LOGD("Detector::initBow detection file read error exit");
        return;
    }

    //LOGD("Detector::initBow Marker Information: ");
    //LOGD("Detector::initBow descriptor type: %s, (width, height) = (%lf, %lf)", descriptor_name.c_str(), this->resizedWidth, this->resizedHeight);

    for (size_t i = 0; i < this->m_db->size(); ++i)
    {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;

        int height, width;
        int oHeight, oWidth;
        std::string file_name;
        char id[50];
        sprintf(id,"%d",i);
        fs["keypoints_" +std::string(id)]>> keypoints;
        fs["descriptors_" + std::string(id)]>> descriptors;
        fs["oWidth_" + std::string(id)] >> oWidth;
        fs["oHeight_" + std::string(id)] >> oHeight;
        fs["file_name_" + std::string(id)] >> file_name;



        TemplateInfo mm;
        mm.descriptors = descriptors;
        mm.keypoints = keypoints;
        mm.itemID = i;
        mm.oHeight = oHeight;
        mm.oWidth = oWidth;
        mm.file_name = file_name;
        fs["templimg_"+std::string(id)] >> mm.templ;
        //LOGD("Detector::initBow Id: %d, keypoints size: %d descriptor size: %d %d templ [ %d %d ] file name %s",
             //mm.itemID, mm.keypoints.size(), mm.descriptors.rows, mm.descriptors.cols,mm.templ.cols,mm.templ.rows,mm.file_name.c_str());

        this->temp_infos.push_back(mm);
    }
}

DBoW3::QueryResults Bow::query(cv::Mat feature) {
    DBoW3::QueryResults ret;
    m_db->query(feature, ret, max_results);
    return ret;
}
bool Bow::in_rect(const cv::KeyPoint& kp,const cv::Rect& rect)
{
    return ((kp.pt.x>=rect.x)&&(kp.pt.x<=(rect.x+rect.width))&&(kp.pt.y>=rect.y)&&(kp.pt.y<=(rect.y+rect.height)));
}
void Bow::split_region(const cv::Rect& rect_to_split,std::vector<cv::Rect>& region_rects,const std::vector<cv::KeyPoint>& keypoints,std::vector<std::vector<cv::KeyPoint> >& region_keypoints)
{

    cv::Rect ct(rect_to_split.x+rect_to_split.width/4,rect_to_split.y+rect_to_split.height/4,rect_to_split.width/2,rect_to_split.height/2);
    cv::Rect lt(rect_to_split.x,rect_to_split.y,rect_to_split.width/2,rect_to_split.height/2);
    cv::Rect rt(rect_to_split.x+rect_to_split.width/2,rect_to_split.y,rect_to_split.width/2,rect_to_split.height/2);
    cv::Rect lb(rect_to_split.x,rect_to_split.y+rect_to_split.height/2,rect_to_split.width/2,rect_to_split.height/2);
    cv::Rect rb(rect_to_split.x+rect_to_split.width/2,rect_to_split.y+rect_to_split.height/2,rect_to_split.width/2,rect_to_split.height/2);

    region_rects={cv::Rect(lt),cv::Rect(rt),cv::Rect(lb),cv::Rect(rb),cv::Rect(ct)};
    region_keypoints.resize(region_rects.size());
    for(int ki=0;ki<keypoints.size();++ki)
    {
        for(int ri=0;ri<region_rects.size();++ri)
        {
            if(in_rect(keypoints[ki],region_rects[ri]))
            {
                region_keypoints[ri].push_back(keypoints[ki]);
            }
        }
    }
}
bool Bow::process_patch(const cv::Mat &inputFrame,std::vector<TemplateInfo>& tempinfos,TemplateInfo& queryitem,int& num_match)
{
    double patchstime=cv::getTickCount();
    //LOGD("arengine bow into process patch");
    tempinfos.clear();
    cv::Mat resizedInputFrame;

    int gkerSize = (int)ceil(0.75*3.0);
    gkerSize += gkerSize % 2 == 0 ? 1 : 0;
    cv::GaussianBlur(inputFrame, resizedInputFrame, cv::Size(gkerSize, gkerSize), 0.75);
    //cv::resize(resizedInputFrame, resizedInputFrame, cvSize(resizedWidth, resizedHeight));

    std::vector<cv::KeyPoint> keypoints;
    this->fdetector->detect(resizedInputFrame,keypoints);
    std::vector<std::vector<cv::KeyPoint>> region_keypoints;
    cv::Rect rect_to_split(0,0,resizedInputFrame.cols,resizedInputFrame.rows);
    std::vector<cv::Rect> region_rects;
    //LOGD("arengine bow after detect");
    split_region(rect_to_split,region_rects,keypoints,region_keypoints);

    cv::Mat drawimg=inputFrame.clone();
    cv::Mat circleimg=inputFrame.clone();
    std::vector<cv::Mat> descriptors(region_keypoints.size());
    for(int ri=0;ri<region_keypoints.size();++ri)
    {
        if(region_keypoints[ri].size()==0)
            continue;
        this->fdetector->compute(resizedInputFrame,region_keypoints[ri],descriptors[ri]);
    }
    //LOGD("arengine bow after compute");
    std::vector<std::pair<double,int>> rets;
    for(int di=0;di<descriptors.size();++di)
    {
        if(descriptors[di].rows<10)
            continue;
        DBoW3::QueryResults ret = this->query(descriptors[di]);
        for(int ri=0;ri<max_results;++ri)
        {
            if(ret[ri].Id>=0&&ret[ri].Id<=99)
                rets.push_back(std::pair<double,int>(ret[ri].Score,int(ret[ri].Id)));
        }
    }
    //LOGD("arengine bow after query");
    std::sort(rets.begin(),rets.end(),[](std::pair<double,int>a,std::pair<double,int> b)
    {
        if(a.second>b.second)return a.second>b.second;
        if(a.second<b.second)return a.second>b.second;
        if(a.second==b.second)return a.first>b.first;
    });
    int num_top=0;
    for(int ri=1;ri<rets.size();++ri)
    {
        if(rets[ri].second!=rets[num_top].second)
        {
            num_top++;
            rets[num_top]=rets[ri];
        }
    }
    rets.resize(num_top);

    std::sort(rets.begin(),rets.end(),[](std::pair<double,int>a,std::pair<double,int> b){return a.first>b.first;});
    //LOGD("arengine bow after sort and num_top %d ",num_top);
    rets.resize(num_top);
    int idx=0;
    int num_max_matches=0;
    //double score_max=rets[0].first;
    //LOGD("arengine bow before knnmatch");
    if(rets.size()!=0)
    {
        std::cout<<"max result "<<max_results<<std::endl;
        for(int i=0;i<max_results;++i)
        {
            tempinfos.push_back(temp_infos[rets[i].second]);
            std::cout<<"max result img name is "<<temp_infos[rets[i].second].file_name<<std::endl;
            //get most stable matching from top k
            int matches_size=0;
            for(int di=0;di<descriptors.size();++di)
            {
		std::vector<cv::Point2f> srcpoints,dstpoints;
                std::vector<cv::Point2f> totalpoints;
                for(int ti=0;ti<temp_infos[rets[i].second].keypoints.size();++ti)
                {
                    totalpoints.push_back(temp_infos[rets[i].second].keypoints[ti].pt);
                }
                std::vector<cv::DMatch> good_matches;
                double min_dist=1000;
                std::vector<std::vector<cv::DMatch>> matches1,matches2;
                matcher->knnMatch(descriptors[di],temp_infos[rets[i].second].descriptors,matches1,2);
                int patch_match_size=0;
                for(int mi=0;mi<matches1.size();++mi)
                {
                    
                    if(matches1[mi][0].distance<1./1.5*matches1[mi][1].distance)
                    {
                        good_matches.push_back(matches1[mi][0]);
                        patch_match_size++;
                        srcpoints.push_back(temp_infos[rets[i].second].keypoints[matches1[mi][0].trainIdx].pt);
                        dstpoints.push_back(region_keypoints[di][matches1[mi][0].queryIdx].pt);
                        //cv::circle(drawimg,region_keypoints[di][mi].pt,3,cv::Scalar(255,255,255),4);
                        cv::circle(circleimg,region_keypoints[di][matches1[mi][0].queryIdx].pt,3,cv::Scalar(255,255,255),4);
                    }
                }
                matches_size=matches_size<patch_match_size?patch_match_size:matches_size;
                //LOGD("matches size is %d",matches1.size());
                //draw matches
                cv::Mat srcimg=cv::imread("../movie100/"+temp_infos[rets[i].second].file_name);
                cv::resize(srcimg,srcimg,cv::Size(srcimg.cols*320./srcimg.rows,320)); 
                cv::Mat outimg;
                
                std::cout<<"srcimg size "<<srcimg.cols<<" "<<srcimg.rows<<std::endl;
                std::cout<<"inputimg size "<<inputFrame.cols<<" "<<inputFrame.rows<<std::endl;
                cv::drawMatches(inputFrame,region_keypoints[di],srcimg,temp_infos[rets[i].second].keypoints,good_matches,outimg); 
                if(srcpoints.size()>=9&&dstpoints.size()>=9) 
                {
                    cv::Mat inlier;
                    
                    cv::Mat homo=cv::findHomography(srcpoints,dstpoints,cv::RANSAC,0.5,inlier);
                    std::vector<cv::Point2f> tempsrc;
                    std::vector<cv::Point2f> tempdst;
                    for(int ii=0;ii<srcpoints.size();++ii)
                    {
                        if(inlier.at<uchar>(ii))
                        {
                            tempsrc.push_back(srcpoints[ii]);
                            tempdst.push_back(dstpoints[ii]);
                        }
                    }
                    if(tempsrc.size()!=0&&tempdst.size()!=0)
                        homo=cv::findHomography(tempsrc,tempdst);

                    cv::Mat affineimg(inputFrame.size(),srcimg.type());
                    cv::warpPerspective(srcimg,affineimg,homo,inputFrame.size());  
                    cv::imshow("affineimg",affineimg);
                    cv::waitKey(10);
                    std::vector<cv::Point2f> preflowpts,posflowpts;
                    std::cout<<"totalpoint size "<<srcpoints.size()<<std::endl;
                    cv::perspectiveTransform(srcpoints,preflowpts,homo); 
                    
                    //std::vector<cv::Point2f> nextpoints(preflowpts.begin(),preflowpts.end());
                    std::vector<uchar> status,secstatus;
                    std::vector<float> error,secerror; 
                    cv::calcOpticalFlowPyrLK(affineimg,inputFrame,preflowpts,dstpoints,status,error,cv::Size(21,21),3,cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01),cv::OPTFLOW_USE_INITIAL_FLOW);
                    posflowpts.assign(preflowpts.begin(),preflowpts.end());
                    cv::calcOpticalFlowPyrLK(inputFrame,inputFrame,dstpoints,posflowpts,secstatus,secerror,cv::Size(21,21),3,cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01),cv::OPTFLOW_USE_INITIAL_FLOW);
                    std::vector<cv::Point2f> newsrcpts,newdstpts;
                    for(int si=0;si<status.size();++si)
                    {
                        if(status[si]&&secstatus[si]&&cv::norm(posflowpts[si]-preflowpts[si])<4)
                        {
                            newsrcpts.push_back(srcpoints[si]);
                            newdstpts.push_back(dstpoints[si]);
                        }
                    }
                    
                    std::vector<cv::Point2f> srcbound;
                    srcbound.push_back(cv::Point2f(0,0));
                    srcbound.push_back(cv::Point2f(srcimg.cols-1,0));
                    srcbound.push_back(cv::Point2f(srcimg.cols-1,srcimg.rows-1));
                    srcbound.push_back(cv::Point2f(0,srcimg.rows-1));
                    std::vector<cv::Point2f> dstbound;
                    cv::perspectiveTransform(srcbound,dstbound,homo);
                    std::vector<cv::Point> drawbound(dstbound.begin(),dstbound.end());
                    std::cout<<"outimg size "<<outimg.cols<<" "<<outimg.rows<<std::endl;
                    std::cout<<"src bound"<<srcbound<<std::endl;
                    std::cout<<"drawbound "<<drawbound<<std::endl;
                    cv::polylines(outimg,drawbound,true,cv::Scalar::all(255),5);
                    std::vector<cv::Point2f> newdstbound;
                    if(newsrcpts.size()>=10&&newdstpts.size()>=10)
                    {
                        cv::Mat newhomo=cv::findHomography(newsrcpts,newdstpts);
                        cv::perspectiveTransform(srcbound,newdstbound,newhomo);
                        drawbound.assign(newdstbound.begin(),newdstbound.end()); 
                        cv::polylines(outimg,drawbound,true,cv::Scalar(0,0,255),5);
                    }
                     
                    
                }
                cv::imshow("outimg",outimg);
                cv::waitKey(0);
                
                //--draw matches
            }

            if(num_max_matches<0.9*matches_size)
            {
                num_max_matches=matches_size;
                idx=i;
                //score_max=rets[i].first;
            }
        }
    } else
    {
        return false;
    }
    cv::imshow("circle img",circleimg);
    cv::imshow("keypoints ",drawimg);
    cv::waitKey(0);
    queryitem=temp_infos[rets[idx].second];
    num_match=num_max_matches;
    //LOGD("matches size is %d and time is %f",num_match,((cv::getTickCount()-patchstime)*1000./cv::getTickFrequency()));
    if(num_match>5)
        return true;
    else
        return false;
}
void Bow::process(const cv::Mat &inputFrame,TemplateInfo& tempinfo)
{
    //LOGD("Detector::detect enter");

    cv::Mat resizedInputFrame;

    int gkerSize = (int)ceil(0.75*3.0);
    gkerSize += gkerSize % 2 == 0 ? 1 : 0;
    cv::GaussianBlur(inputFrame, resizedInputFrame, cv::Size(gkerSize, gkerSize), 0.75);
    cv::resize(resizedInputFrame, resizedInputFrame, cvSize(resizedWidth, resizedHeight));
    tempinfo.itemID = -1;
    double detstime=cv::getTickCount();
    this->fdetector->detect(resizedInputFrame,tempinfo.keypoints);
    //LOGD("keyponts size is %d",keypoints.size());
    tempinfo.keypoints.resize(200);
    this->fdetector->compute(resizedInputFrame,tempinfo.keypoints,tempinfo.descriptors);
    //this->fdetector->detectAndCompute(resizedInputFrame, cv::noArray(), keypoints, descriptors);
    double detetime=cv::getTickCount();
    //LOGD("ARSDK::estimateKeypointMatchingHomography find Detector time is %f",((detetime-detstime)*1000./cv::getTickFrequency()));
    if(tempinfo.descriptors.rows<=0)
    {
        //LOGD("Detector::detect not features exit error");
        return ;
    }

    //LOGD("Detector::detect query database");
    double querystime=cv::getTickCount();
    DBoW3::QueryResults ret = this->query(tempinfo.descriptors);
    //LOGD("query Detector time is %f",((cv::getTickCount()-querystime)*1000./cv::getTickFrequency()));
    if(ret.size()<=0)
    {
        //LOGD("Detector::detect no query result exit error");
        return ;
    }
    //LOGD("Detector::detect get query with id: %d", ret[0].Id);
    tempinfo=temp_infos[ret[0].Id];
}
