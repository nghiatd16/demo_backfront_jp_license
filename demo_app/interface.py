from tkinter import *
from tkinter import filedialog
from tkinter import ttk
import _thread
import threading
import time
import logging
import cv2
from PIL import Image, ImageTk
import uuid
import random as rd
from OCR_driver_license import OCR
STATUS_INACTIVE = 0
STATUS_INPROGRESS = 1
STATUS_DONE = 2
STATUS = STATUS_INACTIVE
STATUS_CONFIRM = 3
result = None
msg_result = None
msg_admin_reviewer = None
showing_admin_review = False
db = None


def response_learning_OK(msg, person):
    global STATUS, result, msg_result
    logging.info("{} | person {}".format(msg, person))
    result = person
    msg_result = msg
    STATUS = STATUS_DONE


def response_cancel():
    global STATUS
    STATUS = STATUS_INACTIVE
    result = None


def reset():
    global STATUS
    STATUS = STATUS_INACTIVE
    result = None


def center(win):
    """
   centers a tkinter window
   :param win: the root or Toplevel window to center
   """
    win.update_idletasks()
    width = win.winfo_width()
    frm_width = win.winfo_rootx() - win.winfo_x()
    win_width = width + 2 * frm_width
    height = win.winfo_height()
    titlebar_height = win.winfo_rooty() - win.winfo_y()
    win_height = height + titlebar_height + frm_width
    x = win.winfo_screenwidth() // 2 - win_width // 2
    y = win.winfo_screenheight() // 2 - win_height // 2
    win.geometry('{}x{}+{}+{}'.format(width, height, x, y))
    win.deiconify()

def message_box(msg):
    label = None
    root = Tk()
    root.title("Notification")
    row = Frame(root)
    idCode_lb = Label(row, text=msg, width=min(50, max(len(msg)+10, 30)), font=(16), anchor='w')

    idCode_lb.pack(side=TOP)

    row.pack(side=TOP, fill=X, padx=5, pady=5)
    check = 0
    def call_cancel(event=None):
        response_cancel()
        root.destroy()

    btn_cancel = Button(root, text="OK", width=10, command=call_cancel)
    btn_cancel.pack(side=BOTTOM, padx=5, pady=5)
    root.bind('<Escape>', call_cancel)
    center(root)
    root.mainloop()
license_img_path = ""
def choose_driver_license(ocr_object):
    
    def browsefunc(license_img):
        global license_img_path
        try:
            path = filedialog.askopenfilename()
            img = Image.open(path)
            img = img.resize((800, 600))
            photo = ImageTk.PhotoImage(img)
            license_img.configure(image=photo)
            license_img.image = photo
            license_img_path = path
            # print(license_img_path)
        except Exception as e:
            logging.exception(e)
            license_img_path = ""
            pass
    def call_ok(ocr_object):
        global license_img_path
        # print(license_img_path)
        if license_img_path == "":
            message_box("You haven't chosen image yet !!!")
            return
        img = cv2.imread(license_img_path)
        lst_text, lst_img = ocr_object.OCR_driver_license(img, std_for_debug=True)
        show_result(lst_img, lst_text)
        return
        

    def call_cancel(event=None):
        # response_cancel()
        root.destroy()
    root = Tk()
    root.title("Japanese Driver License OCR")
    im = Image.open("assets/empty.jpg")
    im = im.resize((800, 600))
    # w, h, c = im.shape
    tkimage = ImageTk.PhotoImage(im)
    license_img = Label(root, image=tkimage)
    license_img.pack()
    btn_browse = Button(root, text="Browse", command=lambda: browsefunc(license_img), anchor="w")
    btn_browse.pack()
    btn_ok = Button(root, text="Crop Image", width=10, command=lambda: call_ok(ocr_object), anchor="w")
    btn_ok.pack(side=LEFT, padx=5, pady=5)
    btn_cancel = Button(root, text="Cancel", width=10, command=call_cancel)
    btn_cancel.pack(side=RIGHT, padx=5, pady=5)
    root.bind('<Return>', call_ok)
    root.bind('<Escape>', call_cancel)
    center(root)
    root.mainloop()

def show_result(lst_img=None, lst_text=None):
    assert len(lst_img) == len(lst_text)
    ## Main window
    root = Toplevel()
    ## Grid sizing behavior in window
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)
    ## Canvas
    cnv = Canvas(root, width=1400, height=720)
    cnv.grid(row=0, column=0, sticky='nswe')
    ## Scrollbars for canvas
    hScroll = Scrollbar(root, orient=HORIZONTAL, command=cnv.xview)
    hScroll.grid(row=1, column=0, sticky='we')
    vScroll = Scrollbar(root, orient=VERTICAL, command=cnv.yview)
    vScroll.grid(row=0, column=1, sticky='ns')
    cnv.configure(xscrollcommand=hScroll.set, yscrollcommand=vScroll.set)
    ## Frame in canvas
    frm = Frame(cnv)
    ## This puts the frame in the canvas's scrollable zone
    cnv.create_window(0, 0, window=frm, anchor='nw')
    ## Frame contents
    for idx in range(len(lst_img)):
        im = Image.fromarray(lst_img[idx])
        tkimage = ImageTk.PhotoImage(im)
        myvar = Label(frm, image=tkimage)
        myvar.image = tkimage
        myvar.pack()
        Label(frm, text=lst_text[idx]).pack()

    ## Update display to get correct dimensions
    frm.update_idletasks()
    ## Configure size of canvas's scrollable zone
    cnv.configure(scrollregion=(0, 0, frm.winfo_width(), frm.winfo_height()))
    ## Go!
    center(root)
    root.mainloop()
# show_result()
class ProgressBar(Tk):

    def __init__(self, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)
        self.button = ttk.Button(text="start", command=self.start)
        self.button.pack()
        self.progress = ttk.Progressbar(self, orient="horizontal",
                                        length=200, mode="determinate")
        self.progress.pack()

        self.bytes = 0
        self.maxbytes = 0

    def start(self):
        self.progress["value"] = 0
        self.maxbytes = 50000
        self.progress["maximum"] = 50000
        self.read_bytes()

    def read_bytes(self):
        '''simulate reading 500 bytes; update progress bar'''
        self.bytes += 1000
        self.progress["value"] = self.bytes
        if self.bytes < self.maxbytes:
            # read more bytes after 100 ms
            self.after(100, self.read_bytes)
        else:
            self.destroy()

# app = ProgressBar()
# app.mainloop()
# choose_driver_license()
def run():
    ocr_object = OCR()
    choose_driver_license(ocr_object)