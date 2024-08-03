import numpy as np
import cv2
import sklearn
import matplotlib.pyplot as plt 
import pickle

#Load the built models
haar = cv2.CascadeClassifier('../model/haarcascade_frontalface_default.xml') # cascade classifier
model_svm =  pickle.load(open('../model/model_svm.pickle',mode='rb')) # machine learning model (SVM)
pca_models = pickle.load(open('../model/pca_data.pickle',mode='rb')) # pca dictionary

model_pca = pca_models['pca_model']
mean_face_arr = pca_models['mean']

#Define a function that uses the haarcascade model to detect the face locations of a provided image.
def face_recognition(path):
    #Step1 : Read Image
    img = cv2.imread(path)
    #Step2 : Convert to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #Step3 : Crop the face
    faces = haar.detectMultiScale(gray,1.5,3)
    pred=[]
    for x,y,w,h in faces:
        roi = gray[y:y+h,x:x+w]
        aoi = img[y:y+h,x:x+h]
        #plt.imshow(roi,cmap='gray')
        #plt.show()
        #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),4)
    #Step4: Normalize
        roi = roi/255.0
    #Step5: Resize Image to (100x100)
        if roi.shape[1] > 100:
            roi_resize = cv2.resize(roi,(100,100),cv2.INTER_AREA)
        else:
            roi_resize = cv2.resize(roi,(100,100),cv2.INTER_CUBIC)
        #Step6 Flatten the Image:(1x10000)
        roi_resize = roi_resize.reshape(1,10000)
        #Step7 : Subtract Mean face
        roi_mean = roi_resize - mean_face_arr
        #Step8 : Get Eigen Image - Apply PCA to roi
        eigen_image = model_pca.transform(roi_mean)
        # step-09 Eigen Image for Visualization
        eig_img = model_pca.inverse_transform(eigen_image)
        # step-10: pass to ml model (svm) and get predictions
        results = model_svm.predict(eigen_image)
        prob_score = model_svm.predict_proba(eigen_image)
        prob_score_max = prob_score.max()
        #Step11 - Generate Report
        text = "%s : %d"%(results[0],prob_score_max*100)
            # defining color based on results
        if results[0] == 'Male':
            color= (255,255,0)
        else:
            color=(255,0,255)
        
        #Return the results generated in the form of a dictionary     
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color,-1)
        cv2.putText(img,text,(x,y),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255),2)
        output = {
                'roi':roi,
                'eig_img': eig_img,
                'prediction_name':results[0],
                'score':prob_score_max
                }
            
        pred.append(output)  
    return img,pred,aoi
    