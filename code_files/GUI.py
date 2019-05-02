from tkinter import * #the "*" means imoprt everythin
import os
from tensorflow import keras
from keras.models import load_model
import numpy as np
from PIL import Image
from PIL import ImageOps


class Window(Frame):
    def __init__(self, master, width = 800, height = 600):
        super().__init__(master)
        self.master = master #main window 
        self.w = width
        self.h = height
        self.points = []
        self.single_value = 3

    def hori_crop(self, arr): #crops the acces horizontal pixels
        h = []
        r = arr.shape[1]
        for array in arr:
            if np.sum(array) != self.single_value * r:
                h.append(array)
        return(np.array(h))

    def vert_crop(self, arr): #crops the acces vertical pixels
        r = arr.shape[0]
        v = [[] for i in range(r)]
        for i in range(arr[0].shape[0]):
            if np.sum(arr[:, i]) != self.single_value * r:
                c = 0
                for array in v:
                    array.append(arr[c, i])
                    c += 1
        return(np.array(v))

    def crop(self, arr): #only works on "RGB" array (x,y,3)
        cropped_list = self.hori_crop(arr)
        cropped_list = self.vert_crop(cropped_list)
        return(cropped_list)

    def flat(self, arr): #flattens the data arr
        l = []
        for array in arr:
            l.append(array.flatten())
        print(l)
        return(np.array(l))
    
    def point(self, event): #plots a point when called at the location 
        self.canvas.create_oval(event.x, event.y, event.x+1, event.y+1, fill="black")
        self.points.append(event.x)
        self.points.append(event.y)
        return(self.points)

    def clear(self):
        self.canvas.delete("all")
        self.prediction_text.delete(1.0, END)
        try:
            os.remove("/Users/maor/Documents/src/Digit_Detector/input_pics/pic_0.eps") #deletes the previouse converted pic
            os.remove("/Users/maor/Documents/src/Digit_Detector/input_pics/conv_pic.png") #deletes the previouse converted pic
        except Exception as e:
            print(f"The photo is already deleted({e})")

    def save_canvas(self):
        self.canvas.update()
        self.canvas.postscript(file = "/Users/maor/Documents/src/Digit_Detector/input_pics/pic_0.eps")

    def format_pic(self, picture): #formats the picture to a 28x28 picture so the model can predict it and returns it
        img = Image.open(f"/Users/maor/Documents/src/Digit_Detector/input_pics/{picture}") #opens the image
        img.save("/Users/maor/Documents/src/Digit_Detector/input_pics/conv_pic.png", "png")
        img = Image.open("/Users/maor/Documents/src/Digit_Detector/input_pics/conv_pic.png")
        img_arr = np.asarray(img)
        img_arr = img_arr / 255
        img_arr = self.crop(img_arr)
        bg_img = Image.new("L", (28,28), 255)
        bg_w, bg_h = bg_img.size
        img = Image.fromarray(np.uint8((img_arr * 255)))
        img.thumbnail((28,28))
        img.convert("L")
        img_w, img_h = img.size
        loc = (int(bg_w/2 - img_w/2), int(bg_h/2 - img_h/2)) #location of where the image will be "pasted"
        bg_img.paste(img, loc)
        bg_img = ImageOps.invert(bg_img)
        bg_img.save("/Users/maor/Documents/src/Digit_Detector/input_pics/conv_pic.png")
        return(bg_img)

    def prediction(self):
        self.save_canvas()
        number = self.format_pic("pic_0.eps")
        number = np.asarray(number)
        number = number / 255
        number = number.flatten()
        number = np.array([number])
        model = load_model("num_class_model.h5")
        prediction = model.predict(number) #model.predict ALWAYS takes an array or a list and it always returns the one hot array (probability distributions/ each nurons "confidence" level)
        self.prediction_text.insert(END, np.argmax(prediction)) #inputs the most "confident" nueron's number in the textbox
        
    def set_win(self):
        self.master.title("Number Prediction")
        self.master.geometry(f"{self.w}x{self.h}")
        self.pack(fill = BOTH, expand = 1) #makes sure that is adjusts to the window size
        predict_label = Label(self.master, text = "Prediction") #creating the label that says "prediction"
        self.prediction_text = Text(self.master, bg = "light blue", width = int(self.h / 50), height = int(self.h / 100))
        self.canvas = Canvas(self.master, width = int(self.w / 2), height = int(self.h / 2), bg = "light blue")
        reset_button = Button(self.master, text = "Reset", command = self.clear) #creating the "reset button" and when you press it, it clears the predictionbox and canvas (MAKE SURE NOT TO PASS A FUNCTION WITH "()" TO THE COMMAND!!!)
        predict_button = Button(self.master, text = "Predict", command = self.prediction)
        #placeing the widgets with all with same location (0, 0) and then changing them later
        self.prediction_text.place(x = 0, y = 0)
        predict_label.place(x = 0, y = 0)
        self.canvas.place(x = 0, y = 0)
        reset_button.place(x = 0, y = 0)
        predict_button.place(x = 0, y= 0)
        #updates the wiggets to be able to check dementions, the "reset_button.winfo_width()" always returns 1 without this command
        self.prediction_text.update()
        predict_label.update()
        self.canvas.update()
        reset_button.update()
        predict_button.update()
        #replaces the widgets location with the intended one
        predict_label.place(x = predict_label.winfo_width(), y = int(self.h / 100))
        self.prediction_text.place(x = int(self.w / 13), y = predict_label.winfo_height() * 2)
        self.canvas.place(x = int(self.w / 2) - int(self.canvas.winfo_width() / 2), y = int(self.h / 4))
        self.canvas.bind("<B1-Motion>", self.point) #binds the movement and button click to start plotting points
        reset_button.place(x = (self.w - int(reset_button.winfo_width() * 1.5)), y = int(self.h / 100))
        predict_button.place(x = int(self.w / 2) - int(predict_button.winfo_width() / 2), y = self.h - predict_button.winfo_height() * 2)



if __name__ == "__main__":
    root = Tk() #root of the window
    app = Window(root)
    app.set_win()
    root.mainloop() #generates the window
