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
        self.draw_crop_rect = False
        self.line_len = 20
        self.trigger_draw_h_line = False
        self.trigger_draw_v_line = False
        self.trigger_draw_crop = False
        self.reset_draw = False
        self.area_ratio_low = 10
        self.area_ratio_up = 50
        self.crop_left_top = None

    @staticmethod
    def pad_image(image):
        padding = 20  # Number of pixels to add to all si
        padded_image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        return padded_image

    def resetFlag(self):
        self.drawing_h_line = False
        self.drawing_v_line = False
        self.trigger_draw = False
        self.draw_crop_rect = False

    def MouseCallback(self,event, x, y, flags, param):
        self.my = int(y *self.src_img.shape[0]/self.preview_h)
        self.mx = int(x *self.src_img.shape[1]/self.preview_w)
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.drawing_h_line:
                self.trigger_draw_h_line=True
            if self.drawing_v_line:
                self.trigger_draw_v_line=True
            if self.draw_crop_rect:
                self.trigger_draw_crop=True
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

    def update_area_ratio_low(self,x):
        self.area_ratio_low = x

    def update_area_ratio_up(self,x):
        self.area_ratio_up = x

    def process_image(self,image_path):
        print("load: ",image_path)
        base_name = os.path.basename(image_path)
        self.src_img = cv2.imread(image_path)
        padded_image = self.pad_image(self.src_img)
        padded_image_area = padded_image.shape[0]*padded_image.shape[1] 
        
        to_save=False
        rect_list = []
        crop_list = []

        while True:
            rect_list = []
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

            contours, _ = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for i, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                area = w*h
                area_ratio = round(area/padded_image_area,2)
                if area_ratio> (self.area_ratio_low/100) and area_ratio < (self.area_ratio_up/100):
                    rect_list.append((x,y,x+w,y+h))

            for rect in rect_list:
                cv2.rectangle(padded_image_modified, (rect[0],rect[1]), (rect[2],rect[3]), (255, 0, 0), 5)    
            for rect in crop_list:
                cv2.rectangle(padded_image_modified, (rect[0],rect[1]), (rect[2],rect[3]), (255, 0, 0), 5)  

            thresholded = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)
            if self.draw_crop_rect:
                if not self.crop_left_top:
                    cv2.circle(padded_image_modified, (self.mx,self.my), 10, (0,0,255), thickness=cv2.FILLED)
                    cv2.circle(thresholded, (self.mx,self.my), 10, (0,0,255), thickness=cv2.FILLED)
                    if self.trigger_draw_crop:
                        self.crop_left_top = (self.mx,self.my)
                        self.trigger_draw_crop = False
                else:
                    cv2.rectangle(padded_image_modified,self.crop_left_top, (self.mx,self.my), (255, 0, 0), 5)    
                    if self.trigger_draw_crop:
                        crop_list.append((self.crop_left_top[0],self.crop_left_top[1],self.mx,self.my))
                        self.trigger_draw_crop = False
                        self.draw_crop_rect = False

            padded_image_preview =cv2.resize(padded_image_modified,(self.preview_w,self.preview_h))
            thresholded_preview =cv2.resize(thresholded,(self.preview_w,self.preview_h))
            cv2.imshow("preview",padded_image_preview)
            cv2.imshow("threshold",thresholded_preview)
            key = cv2.waitKey(10)
            if key == ord('q'):
                exit()
            elif key == ord('n'):
                break
            elif key == ord('v'):
                self.drawing_v_line = True
            elif key == ord('h'):
                self.drawing_h_line = True
            elif key == ord('c'):
                self.draw_crop_rect = True
                self.crop_left_top = None
            elif key == ord('r'):
                padded_image = self.pad_image(self.src_img)
                rect_list = []
                crop_list = []
            elif key == ord('s'):
                if not os.path.exists(self.directory_path):
                    os.makedirs(self.directory_path)
                save_cnt = 0
                save_list = rect_list+crop_list
                for rect in save_list:
                    roi = padded_image[rect[1]:rect[3], rect[0]:rect[2]]
                    img_path = os.path.join(self.directory_path,f'{base_name}_#{save_cnt}.jpg')
                    print("save img",img_path)
                    cv2.imwrite(img_path, roi)
                    save_cnt+=1

    def start(self):
        image_files = list(self.image_files)
        cv2.namedWindow("preview")
        cv2.createTrackbar('ARL', 'preview',self.area_ratio_low, 80, self.update_area_ratio_low)
        cv2.createTrackbar('ARU', 'preview', self.area_ratio_up, 100, self.update_area_ratio_up)
        cv2.namedWindow("threshold")
        cv2.createTrackbar('WThres', 'threshold', self.white_thres, 254, self.update_white_thres)
        cv2.setMouseCallback("preview", self.MouseCallback)
        cv2.setMouseCallback("threshold", self.MouseCallback)
        # print(image_files)
        while len(image_files)>0:
            image_path = image_files[0]
            del image_files[0]
            self.process_image(image_path)


if __name__ == "__main__":
    CR = Cropper()
    CR.start()
    cv2.namedWindow("result")

