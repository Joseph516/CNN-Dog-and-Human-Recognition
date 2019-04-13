#!/usr/bin/env python
# coding=UTF-8
'''
@Description: Dog Detector APP GUI 
@Author: Joe
@Verdion: 1.o
'''
from PIL import ImageTk
import PIL.Image
from tkinter import filedialog
from global_path import *
from model_architecture import getBestModelInDir
from keras.models import load_model
from process_function import pathToTensor
import os
import numpy as np
from tkinter import *
import tkinter.messagebox as messagebox


class DogDetector():
    def __init__(self, model_path, label_path):
        self.model = self.setModel(model_path)
        self.labels = self.setLabels(label_path)

    def setModel(self, model_path):
        # load minium loss model
        best_model_path = getBestModelInDir(model_path)
        model = load_model(filepath=best_model_path)
        return model

    def setLabels(self, label_path):
        # load all dog names
        labels = np.load(label_path)
        labels = [label.split('.')[1] for label in labels]
        return labels

    def detector(self, image_path):
        # image process
        tensor = pathToTensor(image_path).astype('float32')/255

        # predict and print result
        pred = np.argmax(self.model.predict(tensor))

        pred_text = "I guess it is a " + self.labels[pred] + " dog!"

        return pred_text


class Application(Frame):
    def __init__(self, model, master=None):
        self.tk = Tk()
        self.frame = Frame(self.tk, bd=2, relief=SUNKEN)
        self.canvas = self.frameInit()
        self.createWidgets()
        self.model = model

    def frameInit(self):
        # setting up a tkinter canvas with scrollbars
        self.frame.grid_rowconfigure(0, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)
        xscroll = Scrollbar(self.frame, orient=HORIZONTAL)
        xscroll.grid(row=1, column=0, sticky=E+W)
        yscroll = Scrollbar(self.frame)
        yscroll.grid(row=0, column=1, sticky=N+S)
        canvas = Canvas(self.frame, bd=0, xscrollcommand=xscroll.set,
                        yscrollcommand=yscroll.set)
        canvas.grid(row=0, column=0, sticky=N+S+E+W)
        xscroll.config(command=canvas.xview)
        yscroll.config(command=canvas.yview)
        self.frame.pack(fill=BOTH, expand=1)
        return canvas

    def createWidgets(self):
        self.alertButton = Button(
            self.tk, text='choose', command=self.guiDetector)
        self.alertButton.pack()

    def guiDetector(self):
        File = filedialog.askopenfilename(
            parent=self.tk, title='Choose an image.')
        if(imageJudge(File) == False):
            return 0
        filename = ImageTk.PhotoImage(PIL.Image.open(File))
        self.canvas.image = filename  # <--- keep reference of your image
        self.canvas.create_image(0, 0, anchor='nw', image=filename)

        # image need to be detected
        pred_text = self.model.detector(File)
        messagebox.showinfo("Result", pred_text)


def imageJudge(fname):
    image_name_array = ['jpg', 'png', 'jpeg']
    suffix = fname.split('.')[-1].lower()
    if suffix not in image_name_array:
        warning_text = "It's not a image, Please choose a image!"
        messagebox.showinfo("Warning", warning_text)
        return False
    else:
        return True


if __name__ == "__main__":
    model = DogDetector(model_path=PATH_DOG_MODEL,
                        label_path='./data/dog_names.npy')

    # run app
    app = Application(model)

    # set window title:
    app.tk.title("Dog Detector App")

    # process loop:
    app.mainloop()
