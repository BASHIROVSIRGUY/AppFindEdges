# -*- coding: utf-8 -*-

from PIL import Image, ImageTk, ImageFilter
from tkinter import Tk, filedialog, Button, Canvas, BOTH
from tkinter.ttk import Frame
from skimage.color import rgb2gray
import numpy as np
import sys
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from skimage.filters import roberts, sobel  # , threshold_otsu, threshold_local
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt
import csv
from math import ceil, floor


"""
Создание класса для нерйронной сети Хемминга 
"""

class Hemming:

    def __init__(self, samples):
        self.N, self.M = np.shape(samples)
        self.samples = samples

    def activation_function(self, x):
        if x < 0:
            f = 0
        elif x > self.N / 2:
            f = self.N / 2
        else:
            f = x
        return f

    def FirstLayer(self):
        T = self.N / 2
        Y = np.array([sum(0.5 * self.samples[i, :] * self.input_word) + T for i in range(self.N)])
        return Y

    def SecondLayer(self, Y):
        S = Y
        eps = 1 / self.M
        deltaY = 2
        while deltaY > 0.1:
            previous_Y = Y
            for j in range(self.N):
                S[j] = Y[j] - eps * (sum(Y) - Y[j])
                Y[j] = self.activation_function(S[j])
            deltaY = sum(list(set(previous_Y) - set(Y)))
        return self.calculate_hemming_distance(Y)

    def calculate_hemming_distance(self, Y):
        Y_max = max(Y)
        num_Y_max = np.array([i for i in range(len(Y)) if Y[i] == Y_max])
        min_dist = self.M
        num_min_dist = -1
        for i in num_Y_max:
            D = self.samples[i] * self.input_word
            hem_dist = len(D[D == -1])
            if hem_dist <= min_dist:
                min_dist = hem_dist
                num_min_dist = i
        return num_min_dist

    def get_num_dict_word(self, input_word):
        self.input_word = input_word
        Y = self.FirstLayer()
        return self.SecondLayer(Y)


"""
Класс для создания приложения 
"""

class Example(Frame):

    def __init__(self):
        super().__init__()
        self.initUI()

        # self.learn_grid()

        self.loadImage()
        self.setGeometry()
        self.canvas_create()

    def initUI(self):
        self.master.title("APP")
        self.master.minsize(128, 128)
        self.master.bind('<Button-3>', self.exit)
        self.pack()

    def loadImage(self):
        path = filedialog.askopenfilename()
        if path:
            self.img = Image.open(path)
        else:
            self.master.destroy()
            print("Unable to load image")

    def setGeometry(self):
        w, h = self.img.size
        ws, hs = self.winfo_screenwidth(), self.winfo_screenheight()
        if w > ws or h > hs:
            self.master.attributes('-fullscreen', True)
            self.resizeImage()

        w, h = self.img.size
        self.master.attributes('-fullscreen', False)
        self.master.geometry(("%dx%d+%d+%d") % (w, h, ws / 2 - w / 2, hs / 2 - h / 2))

    def canvas_create(self):
        w, h = self.img.size

        canvas = Canvas(self,
                        width=w, height=h,
                        bg='white',
                        cursor="pencil")
        tk_img = ImageTk.PhotoImage(self.img)
        canvas.image = tk_img
        self.canvas_img = canvas.create_image(w / 2, h / 2, image=tk_img)
        self.oval = canvas.create_oval(-40, -40, 0, 0)
        canvas.pack(fill=BOTH, expand=True)

        canvas.bind('<Motion>', self.show_objective)
        canvas.bind('<Leave>', lambda e: canvas.delete(self.oval))
        canvas.bind('<Button-1>', self.get_contour)

        canvas.bind('<Button-3>', self.exit)

        self.canvas = canvas

        self.bind('<Configure>', self.resizeFrames)

    def resizeFrames(self, event):
        self.resizeImage()

        new_img = ImageTk.PhotoImage(self.img)
        wi, hi = self.img.size
        self.canvas.delete(self.canvas_img)
        self.canvas.config(width=wi, height=hi)
        self.canvas.image = new_img
        self.canvas.create_image(wi / 2, hi / 2, image=new_img)

    def resizeImage(self):
        ws, hs = self.master.winfo_width(), self.master.winfo_height()
        wi, hi = self.img.size
        if wi > ws or hi > hs:
            k = min(min(wi, ws) / wi, min(hi, hs) / hi)
        else:
            k = min(max(wi, ws) / wi, max(hi, hs) / hi)
        size = (int(k * wi), int(k * hi))
        self.img = self.img.resize(size)

    def show_objective(self, event):
        self.canvas.delete(self.oval)
        self.oval = self.canvas.create_oval(event.x - 12, event.y - 13,
                                            event.x + 13, event.y + 12)

    def black(self, event):
        event.widget.config(background="#000000")
        for r in range(25):
            for c in range(25):
                if self.can[r][c] == event.widget:
                    self.learn_x[r, c] = -1

    def learn_grid(self):

        # a = np.array([-1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1])

        self.learn_x = np.ones((25, 25), dtype=int)
        self.can = [[Canvas(self, width=16, height=16, background="#ffffff") for c in range(25)] for r in range(25)]
        for r in range(25):
            for c in range(25):
                self.can[r][c].grid(row=r, column=c)

                # if a[25*r+c] == -1: self.can[r][c].config(background="#000000")

                self.can[r][c].bind('<Motion>', self.black)
                self.can[r][c].bind('<Leave>', self.black)

    def load_learn_x(self):
        n = 25

        def side_diagonal(x):
            result = np.zeros((n, n), dtype=int)
            for i in range(n):
                for j in range(n):
                    result[i, j] = x[n - j - 1, n - i - 1]
                    result[n - j - 1, n - i - 1] = x[i, j]
            return result

        def middle(x):
            result = np.zeros((n, n), dtype=int)
            fl = floor(n / 2)
            if fl != ceil(n / 2): result[:, fl] = x[:, fl]
            for i in range(n):
                for j in range(floor(n / 2)):
                    result[i, j] = x[i, n - j - 1]
                    result[i, n - j - 1] = x[i, j]
            return result

        def wrt_row(writer, x):
            writer.writerow(x.ravel())
            writer.writerow(x.ravel() * (-1))

        def write_words(writer):
            wrt_row(writer, self.learn_x)
            wrt_row(writer, side_diagonal(self.learn_x))
            wrt_row(writer, self.learn_x.T)
            wrt_row(writer, side_diagonal(self.learn_x).T)

        with open("data.csv", 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')

            write_words(writer)
            self.learn_x = middle(self.learn_x)
            write_words(writer)

    def get_contour(self, event):

        n = 25
        box_img = self.img.crop((event.x - ceil(n / 2), event.y - ceil(n / 2),
                                 event.x + floor(n / 2), event.y + floor(n / 2)))

        box_img.save("box_img.jpg")
        box_img = box_img.convert('HSV')
        arr_im = np.array(box_img)
        data = np.zeros((5, n ** 2))

        for i in range(3):
            data[i] = arr_im[:, :, i].ravel()

        data[3] = np.array([[i for j in range(n)] for i in range(n)]).ravel()
        data[4] = np.array([[range(n)] for i in range(n)]).ravel()

        data = data.T
        data_scaled = StandardScaler().fit_transform(data)
        data_normalized = normalize(data_scaled)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(data_normalized)
        input_word = kmeans.labels_
        plt.imshow(input_word.reshape((n, n)))
        input_word[input_word == 0] = -1

        with open("data.csv", 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            w = []
            for row in csvreader:
                w.append(list(map(int, row)))
            w = np.array(w)

        mrHem = Hemming(w)
        num = mrHem.get_num_dict_word(input_word)

        smpl_mtrx = w[num].reshape(n, n)

        top = smpl_mtrx[0, :]
        left = smpl_mtrx[:, 0]
        bottom = smpl_mtrx[n - 1, :]
        right = smpl_mtrx[:, n - 1]

        x = []
        y = []

        for i in range(n - 1):
            if top[i + 1] != top[i]:
                y.append(0)
                x.append(i)
            if left[i + 1] != left[i]:
                y.append(i)
                x.append(0)
            if bottom[i + 1] != bottom[i]:
                y.append(n)
                x.append(i)
            if right[i + 1] != right[i]:
                y.append(i)
                x.append(n)

        self.canvas.create_line(event.x + x[0] - 12, event.y + y[0] - 12, event.x + x[1] - 12, event.y + y[1] - 12)

    #         smpl_mtrx[smpl_mtrx == 1] = 255
    #         smpl_mtrx[smpl_mtrx == -1] = 0
    #         plt.imshow(smpl_mtrx, cmap=plt.cm.gray)

    def exit(self, event):
        # self.load_learn_x()
        self.master.destroy()


# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#

def main():
    root = Tk()
    ex = Example()
    root.mainloop()


if __name__ == '__main__':
    main()
