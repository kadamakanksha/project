from tkinter import *
import tkinter as tk


from PIL import Image ,ImageTk

from tkinter.ttk import *
from pymsgbox import *


root=tk.Tk()

root.title("3D Hand Geometry Using Machine Learning")
w = tk.Label(root, text="3D Hand Geometry Using Machine Learning",background="gray51",foreground="orange",width=40,height=2,font=(('Forte', 25, 'italic')))
w.place(x=0,y=10)



w,h = root.winfo_screenwidth(),root.winfo_screenheight()
root.geometry("%dx%d+0+0"%(w,h))
root.configure(background="gray51")


from tkinter import messagebox as ms



def right():
    from subprocess import call
    call(["python","Left_Hand_Geometry.py"])

#def right():
 #   from subprocess import call
 #   call(["python","Right_Hand_Geometry.py"])


bg = Image.open(r"R.jpg")
bg.resize((1500,200),Image.ANTIALIAS)
print(w,h)
bg_img = ImageTk.PhotoImage(bg)
bg_lbl = tk.Label(root,image=bg_img)
bg_lbl.place(x=0,y=93)


img=ImageTk.PhotoImage(Image.open("R.jpg"))

img2=ImageTk.PhotoImage(Image.open("R1.jpg"))

img3=ImageTk.PhotoImage(Image.open("R2.jpg"))


logo_label=tk.Label()
logo_label.place(x=0,y=95)



# using recursion to slide to next image
x = 1

# function to change to next image
def move():
	global x
	if x == 4:
		x = 1
	if x == 1:
		logo_label.config(image=img,width=1800,height=700)
	elif x == 2:
		logo_label.config(image=img2,width=1800,height=700)
	elif x == 3:
		logo_label.config(image=img3,width=1800,height=700)
	x = x+1
	root.after(2000, move)

# calling the function
move()


wlcm=tk.Label(root,text="......Welcome to 3D Hand Geometry System ......",width=100,bd=0,height=2,background="gray51",foreground="orange",font=("Times new roman",22,"bold"))
wlcm.place(x=0,y=775)


#Disease2=tk.Button(root,text="Left Hand Geometry",command=left,width=20,bd=0,height=2,background="skyblue",foreground="black",font=("times new roman",14,"bold"))
#Disease2.place(x=950,y=18)


Disease3=tk.Button(root,text="Capture Hand Geometry",command=right,width=20,bd=0,height=2,background="gray51",foreground="orange",font=("Times new roman",14,"bold"))
Disease3.place(x=1100,y=18)


root.mainloop()
