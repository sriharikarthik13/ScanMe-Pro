"""Building the flask app for displaying the Scan Pro application"""
import os
from unittest import result
#from cv2.gapi import crop
import random
import cv2
import matplotlib.image as matimage
import segno
from face_recognitionmodel import face_recognition
from spacy_ner import extract_text
from flask_wtf import FlaskForm
from wtforms import SubmitField,FileField,StringField
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask import Flask, render_template, request



#Setting up the folders to store the inputted images
UPLOAD_FOLDER = 'static/upload'

BUSINESS_UPLOAD_FOLDER = 'static/businesscardsuploaded'



app = Flask(__name__)
app.config['SECRET_KEY']='secret'
basedir = os.path.abspath(os.path.dirname(__file__))
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(basedir, "data.sqlite")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)
Migrate(app,db)

#SQL DB section
#Creating the Table Model:
# This model stores the name,phonenumber, email, designation and gender of the employee
class UserDataModel1(db.Model):
    __tablename__ = "userdata"

    id = db.Column(db.Integer,primary_key=True,autoincrement=True)
    name = db.Column(db.Text)
    employeeid = db.Column(db.Text,unique=True)
    phonenumber = db.Column(db.Text)
    email = db.Column(db.Text)
    designation = db.Column(db.Text)
    gender = db.Column(db.Text)

    def __init__(self,name,employeeid,phonenumber,email,designation,gender):
        self.gender = gender
        self.name = name
        self.employeeid = employeeid
        self.phonenumber = phonenumber
        self.email = email
        self.designation = designation

    def __repr__(self):
        return f"{[self.id,self.name,self.employeeid,self.phonenumber,self.email,self.designation]}"
    


#Creating the forms required to input the businesscard and the user profile image

#Form for the business card
class ImageForm(FlaskForm):
    imageinput = FileField('Image')
    businesscardinput = FileField('Image')
    submit = SubmitField('Submit')
#Form for the user image
class IDForm(FlaskForm):
    userinput = StringField('UserInput')
    userbgroupinput = StringField('UserBGroupInput')
    submit = SubmitField('Submit')

#Building the flask views that utilize the already build face detection and text extraction models to predict the gender of provided image and also extract the information from a business card and genertae an ID card for an employee and also store their details like (NAME,EMAIL,DESIGNATION,PHONE NUMBER) in a database for further reference.
@app.route('/', methods=['GET', 'POST'])
def index():
    form = ImageForm()
    if form.validate_on_submit():
        #Retreive the input data
        GenderImage = form.imageinput.data
        BusinessCardImage = form.businesscardinput.data
        
        print(GenderImage.filename)
        print(BusinessCardImage.filename)
        #Save it in a local upload folder
        path = os.path.join(UPLOAD_FOLDER,GenderImage.filename)
        GenderImage.save(path)
        #Run the Face Detection model on the profile picture stored image
        image,preds,imgactual = face_recognition(path)

        path = os.path.join(BUSINESS_UPLOAD_FOLDER,BusinessCardImage.filename)
        
        BusinessCardImage.save(path)
        #Run the NER model on the stored image
        listextractvals = extract_text(path)

        name = listextractvals[0]
        phonenumber = listextractvals[1]
        email = listextractvals[2]
        designation = listextractvals[3]

        listofemployeeid=[]
        #Store the extracted data in the DB

        valuesofemployeeid = UserDataModel1.query.with_entities(UserDataModel1.employeeid).all()

        for i in valuesofemployeeid:
            listofemployeeid.append(i)

        #Generate a random EmployeeID for each employee
        employeeidval = "0"
        while employeeidval not in listofemployeeid:
            random_num = random.randint(10000,99999)
            if random_num not in listofemployeeid:
                employeeidval = str(random_num)
                break
            

        print(name,phonenumber,email,designation)


        #Run the profile image of the user through the face detection model and store the result in the DB
        # Store the Gender and Score generated for each image
        imagename = "predicted_image.jpg"
        cv2.imwrite(f"./static/saved/{GenderImage.filename}",image)
        cv2.imwrite(f"./static/predict/{imagename}",image)
        cv2.imwrite(f"./static/cropped/{GenderImage.filename}",imgactual)
        results=[]
        for i,obj in enumerate(preds):
            crop_face = obj['roi']
            eigen_face = obj['eig_img'].reshape(100,100)
            gender_name = obj['prediction_name']
            accuracy = obj['score']

            crop_face_loc = f"face_{i}.jpg"
            eigen_face_loc = f"eigen_face_{i}.jpg"
            
            matimage.imsave(f'./static/predict/{crop_face_loc}',crop_face,cmap='gray')
            matimage.imsave(f'./static/predict/{eigen_face_loc}',eigen_face,cmap='gray')

            results.append([gender_name,accuracy])
            ##Code to get the extracted text:

            if gender_name == 'Male':
                user = UserDataModel1(name=name,employeeid=employeeidval,phonenumber=phonenumber,email=email,designation=designation,gender=gender_name)
                print('HELLO')
            elif gender_name == 'Female':
                user = UserDataModel1(name=name,employeeid=employeeidval,phonenumber=phonenumber,email=email,designation=designation,gender=gender_name)
                print("BYE")
            else:
                user = UserDataModel1(name=name,employeeid=employeeidval,phonenumber=phonenumber,email=email,designation=designation,gender="NO DATA")
                print("Issue")
        print(results)
        db.session.add(user)
        db.session.commit()
        
        print(user.id)
        print(user.name)
        print(user.phonenumber)
        print(user.email)
        print(user.designation)
        print(user.gender)


        return render_template("test.html",preds=preds,imagepred=GenderImage.filename)
    return render_template("index.html",form=form)

#building the flask view that extracts the details of a paticular employee based on user input and generate an ID Card for the employee with an additional feature of storing their details in the form of a qr code.
@app.route('/GenerateID',methods=['GET','POST'])
def GenerateID():
    db.create_all()
    form = IDForm()
    #Extract the input data
    if form.validate_on_submit():
        value1 = form.userinput.data
        value2 = form.userbgroupinput.data
        #Query the particular data from the DB
        if "@" in value1:
            user_val = UserDataModel1.query.filter_by(email=value1).first()
        else:
            user_val = UserDataModel1.query.filter_by(name=value1).first()

        #Store the data in named variables
        name = user_val.name
        employeeid = user_val.employeeid
        phonenumber = user_val.phonenumber
        email = user_val.email
        designation = user_val.designation
        gender= user_val.gender
        #Create a dictionary to store the variables
        uservaluedata = f"Employee ID : {employeeid}  \nPhone : {phonenumber} \nEmail : {email} \nGender : {gender}"
        #Generate the QR CODE
        qrcode = segno.make_qr(uservaluedata)

        qrcode.save(f"./static/QR/{name}_QR.png")

        namepath = f"./static/cropped/{name}.jpg"
        qrpath = f"./static/QR/{name}_QR.png"
        #display the ID card on the webpage
        return render_template("idcard.html",namepath=namepath,qrpath=qrpath,name=name,designation=designation,value2=value2)
    


    return render_template("idcardhome.html",form = form)
    


if __name__ == '__main__':
   app.run(debug=True)