from tkinter import filedialog as fd
from tkinter import *
from tkinter import ttk
import tkinter as tk
import lsb
import DWT
from DCT import DCT
from PVD import pvd_lib
import numpy as np
import cv2
from numpy import asarray
from PIL import Image, ImageTk
import os

alpha = 0.45

window = tk.Tk()
# TileBar
window.title("Image Embedder")
window.geometry('900x700')
TAB_CONTROL = ttk.Notebook(window)
TAB1 = ttk.Frame(TAB_CONTROL)
TAB_CONTROL.add(TAB1, text='Embed')
TAB2 = ttk.Frame(TAB_CONTROL)
TAB_CONTROL.add(TAB2, text='Extract')
TAB_CONTROL.pack(expand=1, fill="both")

embedInputFile = ""
outFolder = ""

def openImageFileChooser():
    global embedInputFile
    embedInputFile = fd.askopenfilename()

    lim = Image.open(embedInputFile)
    lim = lim.resize((100, 75), Image.LANCZOS)
    lphoto = ImageTk.PhotoImage(lim)

    emImage['image'] = ""
    inImage['image'] = lphoto
    lphoto.image = lphoto

def getSecretContent():
    # returns secret content bytes, that will be embedded into the carrier
    if contentTypeVar.get() == "Plain text":
        return inputText.get(1.0, "end-1c")

def embedImage():
    algo = variable.get()
    inp = getSecretContent()

    out_f, out_ext = os.path.splitext(os.path.basename(embedInputFile))
    out_f = out_f + '_' + algo + '_image' + ".png"
    fileName = os.path.join(outFolder, out_f)

    if algo == "LSB":
        lsb.encodeImage(embedInputFile, inp, fileName)
    elif algo == "DWT":
        DWT.dwtenc(embedInputFile, inp, fileName)
    elif algo == "PVD":
        pvd_lib.embed_data(embedInputFile, inp, fileName)
    elif algo == "DCT":
        dct_img = cv2.imread(embedInputFile, cv2.IMREAD_UNCHANGED)
        dct_instance = DCT()
        cv2.imwrite(fileName, dct_instance.dctenc(dct_img, inp))

    lim = Image.open(embedInputFile)
    lim = lim.resize((100, 75), Image.LANCZOS)
    lphoto = ImageTk.PhotoImage(lim)

    inImage['image'] = lphoto
    lphoto.image = lphoto

    oim = Image.open(embedInputFile)
    oim = oim.resize((100, 75), Image.LANCZOS)
    ophoto = ImageTk.PhotoImage(oim)

    emImage['image'] = ophoto
    ophoto.image = ophoto

    embedStatusLbl.configure(text="Success")
    print("Embed Success")

def openExtractImageFileChooser():
    global extractInputFile
    extractInputFile = fd.askopenfilename()

    try:
        eim = Image.open(extractInputFile)
    except Exception as e:
        print(f"Error opening image: {e}")
        return

    eim = eim.resize((100, 75), Image.LANCZOS)
    rphoto = ImageTk.PhotoImage(eim)

    eImage['image'] = rphoto
    rphoto.image = rphoto

def extractData():
    algo = extractAlgo.get()

    out_f, out_ext = os.path.splitext(os.path.basename(extractInputFile))
    out_f = out_f + "_" + algo + ".png"
    extractToFilename = os.path.join(outFolder, out_f)

    if algo == "LSB":
        raw = lsb.decodeImage(extractInputFile)
    elif algo == "DWT":
        raw = DWT.dwtdec(extractInputFile)
        os.remove("dwt_image.npy")
    elif algo == "PVD":
        secret_data_filepath = "op.txt"
        pvd_lib.extract_data(embedInputFile, secret_data_filepath, extractInputFile)
        raw = open(secret_data_filepath, "rb").read().decode("utf-8")
        os.remove(secret_data_filepath)
    elif algo == "DCT":
        dct_img = cv2.imread(extractInputFile, cv2.IMREAD_UNCHANGED)
        dct_instance = DCT()
        raw = dct_instance.dctdec(dct_img)

    eim = Image.open(extractInputFile)
    eim = eim.resize((100, 75), Image.LANCZOS)
    rphoto = ImageTk.PhotoImage(eim)

    rightImage['image'] = rphoto
    rphoto.image = rphoto

    print(len(raw))

    if extractContentType.get() == "Plain text":
        extractStatusLbl.configure(text="Success")
        outputTexArea.delete('1.0', "end")
        outputTexArea.insert('1.0', raw)
    elif extractContentType.get() == "File":
        try:
            with open(extractToFilename, "w") as target:
                target.write(raw)
        except Exception as e:
            with open(extractToFilename, "wb") as target:
                target.write(raw)
        print(f"Successfully extracted to: {extractToFilename}")
    else:
        raise AssertionError("Invalid extract content type")

def selectFolder():
    global outFolder
    outFolder = fd.askdirectory()

def onSelectAlgo(*args):
    emImage['image'] = ""

def onContentTypeChange(*args):
    if contentTypeVar.get() == "Plain text":
        inputText.grid(column=1, row=2, sticky=tk.N)
    elif contentTypeVar.get() == "File":
        inputText.grid_forget()
    else:
        raise AssertionError("Unknown selected secret content type: " + contentTypeVar.get())

# Embed Section Start
# Section Header
textLbl = Label(TAB1, text="Image")
textLbl.grid(column=0, row=0, sticky=tk.W)

textLbl = Label(TAB1, text="Secret content")
textLbl.grid(column=1, row=0, sticky=tk.W)

textLbl = Label(TAB1, text="Algorithm")
textLbl.grid(column=2, row=0, sticky=tk.W)

textLbl = Label(TAB1, text="OutFile")
textLbl.grid(column=3, row=0, sticky=tk.W)

# Input Image File Chooser
addBtn = Button(TAB1, text='Select Image', command=openImageFileChooser)
addBtn.grid(column=0, row=1, sticky=tk.N)

# Choose to embed either text or file
choices = ['Plain text', 'File']
contentTypeVar = StringVar(TAB1)
contentTypeVar.set('Plain text')

contentTypeVar.trace_add('write', onContentTypeChange)

secretContentType = OptionMenu(TAB1, contentTypeVar, *choices)
secretContentType.grid(column=1, row=1)

# Input Text Area
inputText = Text(TAB1, height=20, width=40)
inputText.grid(column=1, row=2, sticky=tk.N)

# Select Algorithm
OPTIONS = ["LSB", "DWT", "PVD", "DCT"]  # etc

variable = StringVar(TAB1)
variable.set(OPTIONS[0])  # default value

algorithmMenu = OptionMenu(TAB1, variable, *OPTIONS, command=onSelectAlgo)
algorithmMenu.grid(column=2, row=1, sticky=tk.N)

inImage = Label(TAB1)
inImage.grid(column=0, row=2)

# Folder File Chooser
addBtn = Button(TAB1, text='Select Dir', command=selectFolder)
addBtn.grid(column=3, row=1, sticky=tk.N)

# Embed Button
addBtn = Button(TAB1, text='Embed', command=embedImage)
addBtn.grid(column=4, row=1, sticky=tk.N)

# Success Message
embedStatusLbl = Label(TAB1, text="")
embedStatusLbl.grid(column=5, row=1, sticky=tk.N)

emImage = Label(TAB1)
emImage.grid(column=3, row=2)

# Embed Section END

# Extract Data Section Start
# Extract File
addBtn = Button(TAB2, text='Select File', command=openExtractImageFileChooser)
addBtn.grid(column=0, row=0, sticky=tk.W)

extractContentType = StringVar(TAB2)
extractContentType.set(choices[0])  # default value

def onExtractContentTypeChange(*args):
    if extractContentType.get() == "File":
        outputTexArea.grid_forget()
        extractedDataLbl.grid_forget()
        extractToLabel.grid(column=0, row=2, sticky=tk.NW)
        extractToSelectBtn.grid(column=0, row=3, sticky=tk.NW)
    elif extractContentType.get() == "Plain text":
        extractToLabel.grid_forget()
        extractToSelectBtn.grid_forget()
        outputTexArea.grid(column=0, row=3, sticky=tk.N)
        extractedDataLbl.grid(column=0, row=2, sticky=tk.W)

extractTypeMenu = OptionMenu(TAB2, extractContentType, *choices, command=onExtractContentTypeChange)
extractTypeMenu.grid(column=0, row=1, sticky=tk.NW)

extractToLabel = Label(TAB2, text="Extract to:")

def selectExtractToFile():
    global extractToFilename
    extractToFilename = fd.asksaveasfilename()
    extractToLabel.config(text="Extract to: " + extractToFilename)

extractToSelectBtn = Button(TAB2, text="Select file", command=selectExtractToFile)

extractAlgo = StringVar(TAB2)
extractAlgo.set(OPTIONS[0])  # default value

algorithmMenu = OptionMenu(TAB2, extractAlgo, *OPTIONS)
algorithmMenu.grid(column=1, row=0, sticky=tk.N)

# Extract Button
addBtn = Button(TAB2, text='Extract', command=extractData)
addBtn.grid(column=2, row=0, sticky=tk.W)

extractStatusLbl = Label(TAB2, text="")
extractStatusLbl.grid(column=3, row=0, sticky=tk.W)

eImage = Label(TAB2)
eImage.grid(column=0, row=1)

rightImage = Label(TAB2)
rightImage.grid(column=1, row=1)

extractedDataLbl = Label(TAB2, text="Extracted Data")
extractedDataLbl.grid(column=0, row=2, sticky=tk.W)

# Input Text Area
outputTexArea = Text(TAB2, height=20, width=40)
outputTexArea.grid(column=0, row=3, sticky=tk.N)
# Extract Data Section End

window.mainloop()



