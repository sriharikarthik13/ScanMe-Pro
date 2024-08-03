# Welcome to the ScanMe Pro - ID card generation OPENCV/Spacy Project.


-This project was built to generate employee ID card, by scanning their existing Business cards which has details such as their name,email,designation and phone number and also scans their profile photos and using OpenCV and spacy. This application also stores the employee details in an employee database 

- This project first takes the profile photo image of an user and detects their face in the image using the haar-cascade classifier model and predicts their gender, using a Support Machine Vector model, that has been trained using an input of various images. 

- To build the data for the SVM model, we first perform PCA by utilizing the elbow method and choose the optimal count for PCA components and develop the input data.

- This project also uses a spacy NER detection model that takes the input of various business card images and uses spacy and OPENCV to extract the NER values from the submited business card and store the employee details accordingly.

- After the gender information and employee details are extracted and stored in the database, the user can enter their email address or the employee ID to generate their ID card, which extracts data from the DB and generates an ID card.

- 


# ScanMe Pro
