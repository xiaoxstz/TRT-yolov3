import cv2
import numpy as np
import matplotlib.pyplot as plt
from time import *
import matplotlib.image
from IPython.display import Image
import matplotlib.image as img
from PIL import Image
from PIL import ImageDraw
import os
import pandas as pd

begin_time = time()


blockSize = 31
value = -1

#count = 0  #droplet total number
area = 0  #single droplet area
min_area = 60
max_area = 3500

x_coordinates = []
y_coordinates = []
r_threshold = []

def close(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    iClose = cv2.morphologyEx(image,cv2.MORPH_CLOSE,kernel)
    return iClose


def open(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(16,16))
    iOpen = cv2.morphologyEx(image,cv2.MORPH_OPEN,kernel)
    return iOpen



#find outline
def findConftours(srcImage,binary):
    contours,hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    c_max = []                          #new area fill
    for i in range(len(contours)):   #new area fill
        cnt = contours[i]    #new area fill
        area = cv2.contourArea(cnt)  #new area fill
    #cv2.drawContours(srcImage,contours,-1,(0,0,255),3)      #original outline drawing
    cv2.drawContours(srcImage,c_max,-1,(0,0,255),thickness=-1)  #new area fill ou

#find all fluorescent area and count number
def countAll(contours,image):
    global count
    count = 0
    for i in range(np.size(contours)):
        area = cv2.contourArea(contours[i])  #calculate contour area
        if (area < min_area) or (area > max_area):
            continue
        else:
            (x,y),radius = cv2.minEnclosingCircle(contours[i])
            (x,y,radius) = np.int0((x,y,radius))
            if (740,530) < (x,y) < (1700,1690):
                count = count + 1
            #print(x)
                x_coordinates.append(x)
                y_coordinates.append(y)
                r_threshold.append(radius)
                Img_around = image[y_coordinates[count-1]-(radius+20):y_coordinates[count-1]+(radius+20),x_coordinates[count-1]-(radius+20):x_coordinates[count-1]+(radius+20)]
                width = Img_around.shape[1]
                height = Img_around.shape[0]
                r_temp = []
                g_temp = []
                b_temp = []
                for xi in range(width):
                    for yi in range(height):
                        r = int(Img_around[yi,xi,2])
                        g = int(Img_around[yi,xi,1])
                        b = int(Img_around[yi,xi,0])
                        if r < 20:
                            r_temp.append(r)
                        if g < 20:
                            g_temp.append(g)
                        if b < 20:
                            b_temp.append(b)
                R = np.mean(r_temp)
                G = np.mean(g_temp)
                B = np.mean(b_temp)
                img3 = cv2.circle(image, (x_coordinates[count-1],y_coordinates[count-1]), radius = (r_threshold[count-1]+10), color = (R,G,B), thickness=-1)
                cv2.imwrite('/home/benny/darknet_multiple/data/bacteria_paper_test/augment_save/%0.3d.jpg'% (count), img3)
            #print(x_coordinates[count-1])
            #cv2.circle(image,(x,y),radius,(0,0,255),2)

            #cv2.imshow("temp.jpg",img3)



    #print(x_coordinates[])
    #print(y_coordinates)
    #print(r_threshold)
    return image,count



def cut(img_route,save_route):
    #img = cv2.imread("/home/benny/anaconda3/envs/Histogram/85.jpg")
    #img = cv2.imread("/home/benny/anaconda3/envs/Histogram/yolov3-visulization/86.tif")
    img = cv2.imread(img_route)
    grayImage = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img2 = img.copy()


    #th4 = cv2.adaptiveThreshold(grayImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize, value) #original version
    ret,binImg = cv2.threshold(grayImage, 100, 255, cv2.THRESH_BINARY) #threshold version
    #cv2.imshow('img', grayImage)
    #cv2.imshow('th1', th1)
    #cv2.imshow('th2', th2)
    #cv2.imshow('th3', th3)
    #cv2.imshow('th4', th4)

    #close_image = th4   #original version


    #iClose = close(close_image) #original version
    #cv2.imshow("close",iClose)

    #iOpen = open(iClose)    #original version
    #cv2.imshow("close_and_open",iOpen)

    #contours,hirarchy = cv2.findContours(iOpen,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)   #original version
    contours, hierarchy = cv2.findContours(binImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #threshold version
    #contours,hirarchy = cv2.findContours(th4,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    print("all fluorescent area: "+format(np.size(contours)))
    #show contour

    #res = cv2.drawContours(img,contours,-1,(0,0,255),2)     #draw outline
    res = cv2.drawContours(img,contours,-1,(0,0,255),-1)    #fill the whole area, the change is the last -1
    #tmp = np.zeros(img.shape,np.uint8)
    #res = cv2.drawContours(tmp, contours, -1, (0, 0, 255), 2)
    #plt.imshow(res)
    #plt.show()
    #cv2.imwrite("/home/benny/anaconda3/envs/Histogram/yolov3-visulization/simpleThreshold_86.bmp",res)
    #cv2.imwrite("/home/benny/anaconda3/envs/Histogram/xuee_brightness/crop_save/simpleThreshold_sample1_cycle1-1s_(612).bmp",res)
    cv2.imwrite(save_route+"simpleThreshold_"+".bmp",res)

    #draw minimum circumscribied circle
    res,count = countAll(contours,img2)
    #cv2.imshow("cirle_res",res)
    #plt.imshow(res)
    #plt.show()
    #cv2.imwrite("/home/benny/anaconda3/envs/Histogram/yolov3-visulization/comprehensiveThreshold_86.bmp",res)
    #cv2.imwrite("/home/benny/anaconda3/envs/Histogram/xuee_brightness/crop_save/comprehensiveThrehsold_sample1_cycle1-1s_(612).bmp",res)
    cv2.imwrite(save_route+"ComprehensiveThreh_"+".bmp",res)
    print("the fluorescent number after selecting: "+format(count))

    end_time = time()
    run_time = end_time - begin_time
    print("running time is: ", run_time)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def cut_multiple(images_path,save_route):
    #img = cv2.imread("/home/benny/anaconda3/envs/Histogram/85.jpg")
    #img = cv2.imread("/home/benny/anaconda3/envs/Histogram/yolov3-visulization/86.tif")
    NameList = []
    CountNumber_beforeSelect = []
    CountNumber_afterSelect = []
    for img_item in os.listdir(images_path):
        img_path = os.path.join(images_path, img_item)
        NameList.append(img_item)
        img = cv2.imread(img_path)
        #img = img[530:1760,510:2200]
        grayImage = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img2 = img.copy()
        #th4 = cv2.adaptiveThreshold(grayImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize, value) #original version
        ret,binImg = cv2.threshold(grayImage, 30, 255, cv2.THRESH_BINARY) #threshold version
        #contours,hirarchy = cv2.findContours(iOpen,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)   #original version
        contours, hierarchy = cv2.findContours(binImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #threshold version
        #print(contours)
        #contours,hirarchy = cv2.findContours(th4,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        print("all fluorescent area of "+img_item+" is: "+format(np.size(contours)))
        CountNumber_beforeSelect.append(np.size(contours))
        #simple_num = format(np.size(contours)
        #show contour
        #res = cv2.drawContours(img,contours,-1,(0,0,255),2)     #draw outline
        res = cv2.drawContours(img,contours,-1,(0,0,255),-1)    #fill the whole area, the change is the last -1
        #tmp = np.zeros(img.shape,np.uint8)
        #res = cv2.drawContours(tmp, contours, -1, (0, 0, 255), 2)
        #plt.imshow(res)
        #plt.show()
        #cv2.imwrite("/home/benny/anaconda3/envs/Histogram/yolov3-visulization/simpleThreshold_86.bmp",res)
        #cv2.imwrite("/home/benny/anaconda3/envs/Histogram/xuee_brightness/crop_save/simpleThreshold_sample1_cycle1-1s_(612).bmp",res)
        cv2.imwrite(save_route+"simpleThreshold_"+img_item+".bmp",res)
        #draw minimum circumscribied circle
        res,count = countAll(contours,img2)
        #cv2.imshow("cirle_res",res)
        #plt.imshow(res)
        #plt.show()
        #cv2.imwrite("/home/benny/anaconda3/envs/Histogram/yolov3-visulization/comprehensiveThreshold_86.bmp",res)
        #cv2.imwrite("/home/benny/anaconda3/envs/Histogram/xuee_brightness/crop_save/comprehensiveThrehsold_sample1_cycle1-1s_(612).bmp",res)
        cv2.imwrite(save_route+"ComprehensiveThreh_"+img_item+".bmp",res)
        print("the fluorescent number after selecting: "+img_item+" is: "+format(count))
        CountNumber_afterSelect.append(count)
        #compre_num = format(count)
        #array2.append(compre_num)
        end_time = time()
        run_time = end_time - begin_time
        print("running time is: "+ str(run_time)+"\n")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        file=open('simple_thre_number.txt','w')
        file.write(str(array_simple_th))
        file.close()
        file=open('com_thre_number.txt','w')
        file.write(str(array_comp_th))
        file.close()
        '''
    df=pd.DataFrame(list(zip(NameList,CountNumber_beforeSelect,CountNumber_afterSelect)),columns=['name','SimpleThre','Comprehensive'])
    df.to_excel('/home/benny/darknet_multiple/data/bacteria_paper_test/'+'CountNumber_threshold.xlsx',index=None)
if __name__ == '__main__':
    cut_multiple("/home/benny/darknet_multiple/data/bacteria_paper_test/1/","/home/benny/darknet_multiple/data/bacteria_paper_test/countresult_thre/")
    #cut("/home/benny/anaconda3/envs/Histogram/subtract_result/result45.jpg","/home/benny/anaconda3/envs/Histogram/subtract_result/")
