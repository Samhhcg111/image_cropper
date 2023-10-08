import cv2
import numpy as np
import os
import glob

class Cropper:

    def __init__(self):
        self.directory_path = "output"
        self.input_path = "./input_images/"
        self.image_files = glob.glob(f'{self.input_path}*.jpg')
        self.preview_w = 640
        self.preview_h = 400

        self.white_thres = 200
        self.drawing_h_line = False
        self.drawing_v_line = False
        self.line_len = 20
        self.trigger_draw_h_line = False
        self.trigger_draw_v_line = False
        self.reset_draw = False
    @staticmethod
    def pad_image(image):
        padding = 20  # Number of pixels to add to all si
        padded_image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        return padded_image

    def resetFlag(self):
        self.drawing_h_line = False
        self.drawing_v_line = False
        self.trigger_draw = False

    def MouseCallback(self,event, x, y, flags, param):
        self.my = int(y *self.src_img.shape[0]/self.preview_h)
        self.mx = int(x *self.src_img.shape[1]/self.preview_w)
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.drawing_h_line:
                self.trigger_draw_h_line=True
            if self.drawing_v_line:
                self.trigger_draw_v_line=True
        if event == cv2.EVENT_RBUTTONDOWN:
            self.resetFlag()
        elif event == cv2.EVENT_MOUSEWHEEL:
            if self.drawing_h_line or self.drawing_v_line:
                if flags > 0:
                    self.line_len +=2
                else:
                    self.line_len -=2
                if self.line_len <2:
                    self.line_len = 2

    def update_white_thres(self,x):
        self.white_thres = x

    def process_image(self,image_path):
        print("load: ",image_path)
        self.src_img = cv2.imread(image_path)
        padded_image = self.pad_image(self.src_img)
        padded_image_area = padded_image.shape[0]*padded_image.shape[1] 
        
        to_save=False
        save_cnt = 0
        rect_list = []

        cv2.namedWindow("preview")
        cv2.namedWindow("threshold")
        cv2.createTrackbar('white_thres_up', 'threshold', self.white_thres, 254, self.update_white_thres)
        cv2.setMouseCallback("preview", self.MouseCallback)
        cv2.setMouseCallback("threshold", self.MouseCallback)

        while True:
            padded_image_modified = padded_image.copy()
            if self.drawing_v_line:
                cv2.line(padded_image_modified, (self.mx, 0), (self.mx,padded_image_modified.shape[0]), (255, 255, 255), self.line_len)
                if self.trigger_draw_v_line:
                    cv2.line(padded_image, (self.mx, 0), (self.mx,padded_image_modified.shape[0]), (255, 255, 255), self.line_len)
                    self.trigger_draw_v_line = False

            if self.drawing_h_line:
                cv2.line(padded_image_modified, (0,self.my), (padded_image_modified.shape[1],self.my), (255, 255, 255), self.line_len)
                if self.trigger_draw_h_line:
                    cv2.line(padded_image, (0,self.my), (padded_image_modified.shape[1],self.my), (255, 255, 255), self.line_len)
                    self.trigger_draw_h_line = False

            gray = cv2.cvtColor(padded_image_modified, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            _, thresholded = cv2.threshold(gray, self.white_thres, 255, cv2.THRESH_BINARY)
            kernel = np.ones((5, 5), np.uint8)
            thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN,kernel,iterations=3)

           

            padded_image_preview =cv2.resize(padded_image_modified,(self.preview_w,self.preview_h))
            thresholded_preview =cv2.resize(thresholded,(self.preview_w,self.preview_h))
            cv2.imshow("preview",padded_image_preview)
            cv2.imshow("threshold",thresholded_preview)
            key = cv2.waitKey(1)
            if key == ord('q'):
                exit()
            elif key == ord('n'):
                break
            elif key == ord('v'):
                self.drawing_v_line = True
            elif key == ord('h'):
                self.drawing_h_line = True

    def start(self):
        image_files = list(self.image_files)
        # print(image_files)
        while len(image_files)>0:
            image_path = image_files[0]
            del image_files[0]
            self.process_image(image_path)


if __name__ == "__main__":
    CR = Cropper()
    CR.start()
    cv2.namedWindow("result")

