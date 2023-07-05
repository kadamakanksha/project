# -*- coding: utf-8 -*-ss
"""
Created on Tue May  4 17:28:41 2021

@author: user
"""

from tkinter import *
import tkinter as tk


from PIL import Image ,ImageTk

from tkinter.ttk import *
from pymsgbox import *


root=tk.Tk()

root.title("3D Hand Geometry Using Machine Learning")

#, relwidth=1, relheight=1)

w = tk.Label(root, text="3D Hand Geometry Using Machine Learning",width=100,background="skyblue", foreground="#710F62",height=2,font=("Times new roman",19,"bold"))
w.place(x=0,y=15)



w,h = root.winfo_screenwidth(),root.winfo_screenheight()
root.geometry("%dx%d+0+0"%(w,h))
root.configure(background="skyblue")


from tkinter import messagebox as ms


def Login():
    from subprocess import call
    call(["python","login1.py"])


def Register():
    from subprocess import call
    call(["python","registration.py"])







bg = Image.open(r"h4.jpg")
bg.resize((1500,200),Image.ANTIALIAS)
print(w,h)
bg_img = ImageTk.PhotoImage(bg)
bg_lbl = tk.Label(root,image=bg_img)
bg_lbl.place(x=0,y=93)

wlcm=tk.Label(root,text="......Welcome to 3D Hand Geometry System ......",width=100,height=2,background="skyblue",foreground="#710F62",font=("Times new roman",22,"bold"))
wlcm.place(x=0,y=775)




Disease2=tk.Button(root,text="Login",command=Login,width=15,height=2, bd="0",bg="#FAA911",foreground="white",font=("times new roman",14,"bold"))
Disease2.place(x=10,y=25)


Disease3=tk.Button(root,text="Register",command=Register,width=15, bd="0", height=2, bg="#AF8989",foreground="white",  font=("times new roman",14,"bold"))
Disease3.place(x=200,y=25)


root.mainloop()
