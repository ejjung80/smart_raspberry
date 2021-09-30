from tkinter import *
from tkinter import messagebox
from tensorcam import TensorCam

cam = TensorCam()

def pressed_button() :
    cam.start_recognize(True)

def show_result():
    msg = messagebox.showinfo('result',cam.get_result())

def show_what():
    source = cam.get_result()
    
    ## your code for show the result

    return source

def main():
    try:
        cam.start()

        window = Tk()
        window.geometry('250x250+750+300')
        window.title('TENSOR CAM')

        l = Label(window, bg="white", text='Capturing the Image', font=13)
        l.place(x=30, y=10)
  
        b1 = Button(window, width=12, height=2, text='CAPTURE', bg="magenta", 
                                                command=pressed_button)
        b1.place(x=65, y=60)

        b2 = Button(window, width=12, height=2, text='RESULT', bg="pink", 
                                                command=show_result)
        b2.place(x=65, y=120)

        ## rename the button 
        b3 = Button(window, width=12, height=2, text='WHAT', bg="blue", 
                                                command=show_what)
        b3.place(x=65, y=180)
   
        window.mainloop()

    finally:
        cam.exit()
        cam.join()
        quit()
    return

if __name__ == '__main__':
    main()
