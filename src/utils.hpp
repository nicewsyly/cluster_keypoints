#include <iostream>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <dirent.h>
#include <sys/stat.h>

#define MAX_PATH_LEN (256)

static void list_dir(const char* path, std::vector<std::string>& imgs) {
    DIR *d = NULL;
    struct dirent *dp = NULL; /* readdir函数的返回值就存放在这个结构体中 */
    struct stat st;    
    char p[MAX_PATH_LEN] = {0};
    
    if(stat(path, &st) < 0 || !S_ISDIR(st.st_mode)) {
        printf("invalid path: %s\n", path);
        return;
    }

    if(!(d = opendir(path))) {
        printf("opendir[%s] error: %m\n", path);
        return;
    }

    while((dp = readdir(d)) != NULL) {
        /* 把当前目录.，上一级目录..及隐藏文件都去掉，避免死循环遍历目录 */
        if((!strncmp(dp->d_name, ".", 1)) || (!strncmp(dp->d_name, "..", 2)))
            continue;

        snprintf(p, sizeof(p) - 1, "%s/%s", path, dp->d_name);
        stat(p, &st);
        if(!S_ISDIR(st.st_mode)) {
            //printf("%s\n", dp->d_name);
            imgs.push_back(dp->d_name);
        } else {
            //printf("%s/\n", dp->d_name);
            list_dir(p,imgs);
        }
    }
    closedir(d);

    return;
}
void add_bg(const cv::Mat& bgimg,const cv::Mat foreimg,cv::Mat& miximg)
{
    miximg=bgimg.clone();
    for(int ri=0;ri<miximg.rows;++ri)
    {
        uchar* mixdata=miximg.ptr<uchar>(ri);
        const uchar* foredata=foreimg.ptr<uchar>(ri);
        for(int ci=0;ci<miximg.cols*miximg.channels();ci+=3)
        {
            if(foredata[ci]!=0||foredata[ci+1]!=0||foredata[ci+2]!=0)
            {
                mixdata[ci]=foredata[ci];
                mixdata[ci+1]=foredata[ci+1];
                mixdata[ci+2]=foredata[ci+2];
                
            }
        }
    }
    
}
