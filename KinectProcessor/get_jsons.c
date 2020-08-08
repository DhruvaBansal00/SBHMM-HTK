#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void getJson(const char *mkvVideo, const char *jsonLocation);

void getJson(const char *mkvVideo, const char *jsonLocation) {
    char cmd[10000];
    strcpy(cmd, "./offline_processor ");
    strcat(cmd, mkvVideo);
    strcat(cmd, " ");
    strcat(cmd, jsonLocation);
    system(cmd);
}


int main()
{
    getJson("/home/dhruva/Desktop/CopyCat/Media/matthew_videos/MatthewTest0.mkv", "/home/dhruva/Desktop/CopyCat/Media/kinectJson/temp.json"); 
    return 0;
}