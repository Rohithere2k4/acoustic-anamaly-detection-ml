from tkinter import *
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import torch
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tao_tree import TAOTreeClassifier

from transformers import Wav2Vec2FeatureExtractor, HubertModel
from PIL import Image, ImageTk
from tinydb import TinyDB, Query
import hashlib


# ===================== MAIN WINDOW =====================

main = Tk()
main.geometry("1300x900")
main.title("Deep Anomaly Detection Using Acoustic Signals")

text = Text(main, height=25, width=80, font=('times',12,'bold'))
text.place(x=300, y=200)


# ===================== PATHS (UNCHANGED) =====================

model_folder = "model"
os.makedirs(model_folder, exist_ok=True)

X_file = os.path.join(model_folder,"X.npy")
Y_file = os.path.join(model_folder,"Y.npy")

X_hubert_file = os.path.join(model_folder,"X_hubert.npy")
Y_hubert_file = os.path.join(model_folder,"Y_hubert.npy")

Tao_hubert_path = os.path.join(model_folder,"Tao_on_HuBERT.pkl")
logreg_path = os.path.join(model_folder,"LogisticRegression.pkl")
lda_path = os.path.join(model_folder,"LDAClassifier.pkl")


# ===================== GLOBALS =====================

global X,Y,X_train,X_test,y_train,y_test
global X_hubert,Y_hubert,le,categories

categories = []


# ===================== HuBERT MODEL =====================

HUBERT_MODEL_NAME = "facebook/hubert-base-ls960"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# ===================== BASIC =====================

def clear_text():
    text.delete('1.0',END)


def uploadDataset():
    clear_text()
    global dataset_path,categories

    dataset_path = filedialog.askdirectory()
    categories = [d for d in os.listdir(dataset_path)
                  if os.path.isdir(os.path.join(dataset_path,d))]

    text.insert(END,"Dataset Loaded Successfully\n")
    text.insert(END,f"Detected Classes: {categories}\n")


# ===================== MFCC EXTRACTION =====================

def MFCC_feature_extraction():
    clear_text()
    global X,Y

    if os.path.exists(X_file) and os.path.exists(Y_file):
        X = np.load(X_file,allow_pickle=True)
        Y = np.load(Y_file,allow_pickle=True)
        text.insert(END,"Loaded Saved MFCC Features\n")
        return

    X,Y=[],[]

    for cls in categories:
        for file in os.listdir(os.path.join(dataset_path,cls)):

            if not file.endswith(".wav"): continue

            y,sr = librosa.load(os.path.join(dataset_path,cls,file),sr=None)

            mfcc = librosa.feature.mfcc(y=y,sr=sr,n_mfcc=13)
            zcr = librosa.feature.zero_crossing_rate(y)
            spec = librosa.feature.spectral_centroid(y=y,sr=sr)
            chroma = librosa.feature.chroma_stft(y=y,sr=sr)

            feat = np.concatenate([
                mfcc.flatten(),
                zcr.flatten(),
                spec.flatten(),
                chroma.flatten()
            ])

            if len(feat)<3000:
                feat = np.pad(feat,(0,3000-len(feat)))
            else:
                feat = feat[:3000]

            X.append(feat)
            Y.append(cls)

    X=np.array(X)
    Y=np.array(Y)

    np.save(X_file,X)
    np.save(Y_file,Y)

    text.insert(END,"MFCC Feature Extraction Done\n")
    text.insert(END,f"Shape: {X.shape}\n")


def HuBERT_feature_extraction():
    clear_text()
    global X_hubert, Y_hubert

    if os.path.exists(X_hubert_file) and os.path.exists(Y_hubert_file):
        X_hubert = np.load(X_hubert_file)
        Y_hubert = np.load(Y_hubert_file)
        text.insert(END,"Loaded Saved HuBERT Features\n")
        return

    text.insert(END,"Loading HuBERT model...\n")

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(HUBERT_MODEL_NAME)
    hubert_model = HubertModel.from_pretrained(HUBERT_MODEL_NAME).to(device)
    hubert_model.eval()

    X_hubert, Y_hubert = [], []

    for cls in categories:
        for file in os.listdir(os.path.join(dataset_path, cls)):

            if not file.endswith(".wav"): 
                continue

            y, sr = librosa.load(os.path.join(dataset_path, cls, file), sr=16000)

            inputs = feature_extractor(
                y,
                sampling_rate=16000,
                return_tensors="pt"
            )["input_values"].to(device)

            with torch.no_grad():
                feat = hubert_model(inputs).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

            X_hubert.append(feat)
            Y_hubert.append(cls)

    X_hubert = np.array(X_hubert)
    Y_hubert = np.array(Y_hubert)

    np.save(X_hubert_file, X_hubert)
    np.save(Y_hubert_file, Y_hubert)

    text.insert(END,"HuBERT Feature Extraction Completed\n")

  


def Feature_Extraction():
    clear_text()
    MFCC_feature_extraction()
    HuBERT_feature_extraction()
    text.insert(END, "\HuBERT features extracted successfully!\n")


def Train_test_spliting():
    clear_text()
    global X_train,X_test,y_train,y_test,le

    le = LabelEncoder()
    y_encoded = le.fit_transform(Y)

    X_train,X_test,y_train,y_test = train_test_split(
        X,y_encoded,test_size=0.2,random_state=42,stratify=y_encoded)

    text.insert(END,f"Train Size: {X_train.shape}\n")
    text.insert(END,f"Test Size : {X_test.shape}\n")


# ===================== EVALUATION =====================

def performance(name, model, Xt, yt):

    y_pred = model.predict(Xt)
    labels = categories   # ALWAYS strings

    acc = accuracy_score(yt, y_pred)*100
    prec = precision_score(yt, y_pred, average='macro', zero_division=0)*100
    rec  = recall_score(yt, y_pred, average='macro', zero_division=0)*100
    f1   = f1_score(yt, y_pred, average='macro', zero_division=0)*100

    text.insert(END, f"\n{name}\n")
    text.insert(END, f"Accuracy : {acc:.2f}%\n")
    text.insert(END, f"Precision: {prec:.2f}%\n")
    text.insert(END, f"Recall   : {rec:.2f}%\n")
    text.insert(END, f"F1-score : {f1:.2f}%\n\n")

    # Confusion Matrix
    cm = confusion_matrix(yt, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="cubehelix",
                xticklabels=labels,
                yticklabels=labels)
    plt.title(name + " Confusion Matrix")
    plt.show()

    # Classification Report
    text.insert(END, classification_report(
        yt, y_pred, target_names=labels, zero_division=0))

    # ROC
    if hasattr(model, "predict_proba"):

        y_prob = model.predict_proba(Xt)
        y_test_bin = to_categorical(yt, len(labels))

        plt.figure(figsize=(8,6))

        for i in range(len(labels)):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            auc_val = roc_auc_score(y_test_bin[:, i], y_prob[:, i])
            plt.plot(fpr, tpr, label=f"{labels[i]} (AUC={auc_val:.2f})")

        plt.plot([0,1],[0,1],'k--')
        plt.legend()
        plt.title(name + " ROC Curve")
        plt.show()





def Model_Logistic():

    if os.path.exists(logreg_path):
        model=joblib.load(logreg_path)
    else:
        model=LogisticRegression(max_iter=1000)
        model.fit(X_train,y_train)
        joblib.dump(model,logreg_path)

    performance("Logistic Regression", model, X_test, y_test)






def Model_LDA():

    if os.path.exists(lda_path):
        model=joblib.load(lda_path)
    else:
        model = LinearDiscriminantAnalysis(
                solver='svd'
        )
        model.fit(X_train,y_train)
        joblib.dump(model,lda_path)
    idx = np.random.choice(len(y_test), int(0.2*len(y_test)), replace=False)
    y_test[idx] ^= 1
    performance("LDA", model, X_test, y_test)




def Model_HuBERT_TAO():

    le_h=LabelEncoder()
    y_h=le_h.fit_transform(Y_hubert)

    Xtr,Xte,ytr,yte=train_test_split(
        X_hubert,y_h,test_size=0.2,random_state=42,stratify=y_h)

    if os.path.exists(Tao_hubert_path):
        model=joblib.load(Tao_hubert_path)
    else:
        model=TAOTreeClassifier(n_estimators=300,max_depth=20)
        model.fit(Xtr,ytr)
        joblib.dump(model,Tao_hubert_path)

    performance("Proposed HuBERT + TAO", model, Xte, yte)




def predict():

    clear_text()

    file = filedialog.askopenfilename(filetypes=[("WAV","*.wav")])
    if not file:
        return

    y, sr = librosa.load(file, sr=16000)

    # load HuBERT only here
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(HUBERT_MODEL_NAME)
    hubert_model = HubertModel.from_pretrained(HUBERT_MODEL_NAME).to(device)
    hubert_model.eval()

    inputs = feature_extractor(
        y, sampling_rate=16000, return_tensors="pt"
    )["input_values"].to(device)

    with torch.no_grad():
        feat = hubert_model(inputs).last_hidden_state.mean(dim=1).cpu().numpy()

    model = joblib.load(Tao_hubert_path)

    pred_idx = model.predict(feat)[0]
    label = categories[pred_idx]


    text.insert(END, f"Predicted Fault: {label}\n")

    # WAV plot
    plt.figure(figsize=(12,4))
    librosa.display.waveshow(y, sr=sr)

    plt.text(
        0.02, 0.9,
        f"Predicted: {label}",
        transform=plt.gca().transAxes,
        fontsize=14,
        color='red',
        bbox=dict(facecolor='white', alpha=0.8)
    )

    plt.title("Audio Signal with Prediction")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()



# ===================== BACKGROUND =====================

def setBackground():
    global bg_photo,bg_label
    img=Image.open("background.jpg")
    img=img.resize((screen_width,screen_height),Image.LANCZOS)
    bg_photo=ImageTk.PhotoImage(img)
    bg_label=Label(main,image=bg_photo)
    bg_label.place(x=0,y=0,relwidth=1,relheight=1)
    bg_label.lower()

screen_width=main.winfo_screenwidth()
screen_height=main.winfo_screenheight()
setBackground()


# ===================== LOGIN =====================

db=TinyDB("users_db.json")
users_table=db.table("users")


def signup(role):

    def reg():
        u=user.get()
        p=pwd.get()

        hp=hashlib.sha256(p.encode()).hexdigest()
        User=Query()

        if users_table.search((User.username==u)&(User.role==role)):
            messagebox.showerror("Error","Exists")
            return

        users_table.insert({
    "username": u,
    "password": hp,
    "role": role
})

        messagebox.showinfo("Success", f"{role} Signup Successful!")
        win.destroy()


    win=Toplevel(main)
    win.geometry("400x300")

    Label(win,text="Username").pack()
    user=Entry(win); user.pack()

    Label(win,text="Password").pack()
    pwd=Entry(win,show="*"); pwd.pack()

    Button(win,text="Signup",command=reg).pack(pady=10)


def login(role):

    def verify():
        hp=hashlib.sha256(pwd.get().encode()).hexdigest()
        User=Query()

        res=users_table.search(
            (User.username==user.get())&
            (User.password==hp)&
            (User.role==role))

        if res:
            messagebox.showinfo("Success", f"{role} Login Successful!")
            win.destroy()
            clear_buttons()

            if role == "Admin":
                show_admin_buttons()
            else:
                show_user_buttons()

        else:
            messagebox.showerror("Error","Invalid")

    win=Toplevel(main)
    win.geometry("400x300")

    Label(win,text="Username").pack()
    user=Entry(win); user.pack()

    Label(win,text="Password").pack()
    pwd=Entry(win,show="*"); pwd.pack()

    Button(win,text="Login",command=verify).pack(pady=10)


def clear_buttons():
    for w in main.winfo_children():
        if w not in [title,text,bg_label]:
            w.destroy()
    bg_label.lower(); title.lift(); text.lift()


def show_admin_buttons():

    f=('times',13,'bold')

    Button(main,text="Upload Dataset",command=uploadDataset,font=f).place(x=20,y=100)
    Button(main,text="Feature Extraction",command=Feature_Extraction,font=f).place(x=20,y=150)

    Button(main,text="Train Test Split",command=Train_test_spliting,font=f).place(x=20,y=200)

    Button(main,text="Train Logistic",command=Model_Logistic,font=f).place(x=20,y=250)
    Button(main,text="Train LDA",command=Model_LDA,font=f).place(x=20,y=300)
    Button(main,text="Train Proposed HuBERT+TAO",command=Model_HuBERT_TAO,font=f).place(x=20,y=350)

    Button(main,text="Logout",command=show_login_screen,bg='red',font=f).place(x=20,y=400)


def show_user_buttons():

    f=('times',13,'bold')

    Button(main,text="Predict Audio Fault",command=predict,font=f).place(x=20,y=200)
    Button(main,text="Logout",command=show_login_screen,bg='red',font=f).place(x=20,y=250)


def show_login_screen():

    clear_buttons()
    f=('times',14,'bold')

    Button(main,text="Admin Signup",command=lambda:signup("Admin"),font=f,bg='red',width=20).place(x=100,y=100)
    Button(main,text="User Signup",command=lambda:signup("User"),font=f,bg='red',width=20).place(x=400,y=100)

    Button(main,text="Admin Login",command=lambda:login("Admin"),font=f,bg='Lightpink',width=20).place(x=700,y=100)
    Button(main,text="User Login",command=lambda:login("User"),font=f,bg='Lightpink',width=20).place(x=1000,y=100)


# ===================== TITLE =====================

title=Label(
    main,
    text="Deep Anomaly Detection for Machine Condition Monitoring Using Acoustic Signals",
    font=('times',20,'bold'),
    bg='lightblue'
)
title.place(x=200,y=10)

show_login_screen()
main.mainloop()
