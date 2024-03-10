import cv2 as cv
from moviepy.editor import VideoFileClip, concatenate_videoclips
import os
import numpy as np

class Video:
    def __init__(self, video):
        self.video = video
        self.cap = cv.VideoCapture(video)

    def repeat_clip(self, times):
        clip = VideoFileClip(self.video)
        clips = [clip] * times
        concatenated_clips = concatenate_videoclips(clips)

        base_name = os.path.basename(self.video)
        concatenated_clips.write_videofile(f"assets/extended_{base_name}")

    def get_width(self):
        return int(self.cap.get(3))
    
    def get_height(self):
        return int(self.cap.get(4))
    
    def resize(self, frame):
        return cv.resize(frame, (640, 480))

    def get_current_time(self):
        return int(self.cap.get(cv.CAP_PROP_POS_MSEC))
    
    def between(self, begin, end):
        return begin <= self.get_current_time() < end

    def get_fps(self):
        return int(self.cap.get(5))
    
    def put_subtitles(self, frame, text):
        return cv.putText(frame, text, (50,50), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
    
    def switch_color(self, frame, new_color):
        colors = {
            "GRAY": cv.COLOR_BGR2GRAY,
            "BGR": cv.COLOR_GRAY2BGR
        }
        return cv.cvtColor(frame, colors[new_color])
    
    def gaussian_blur(self, frame, kernel):
        frame = cv.GaussianBlur(frame, (kernel, kernel), 0)
        self.put_subtitles(frame, f"GAUSSIAN with kernel of size {kernel}")
        return frame
    
    def bilateral_filter(self, frame, kernel):
        frame = cv.bilateralFilter(frame, 15, kernel, kernel)
        self.put_subtitles(frame, f"BILATERAL with kernel of size {kernel}")
        return frame
    
    def grab_object_rgb(self, frame):
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        lower_blue = np.array([0, 0, 100])    
        upper_blue = np.array([100, 100, 255]) 

        mask = cv.inRange(rgb, lower_blue, upper_blue)
        return mask
    
    def grab_object_hsv(self, frame):
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        lower_blue = np.array([100, 50, 50])   
        upper_blue = np.array([130, 255, 255]) 

        mask = cv.inRange(hsv, lower_blue, upper_blue)
        return mask
    
    def detect_edges(self, frame, direction):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (3,3), 0)
        sobelx = cv.Sobel(src=blur, ddepth=cv.CV_64F, dx=1, dy=0, ksize=5) 
        sobely = cv.Sobel(src=blur, ddepth=cv.CV_64F, dx=0, dy=1, ksize=5) 
        sobelxy = cv.Sobel(src=blur, ddepth=cv.CV_64F, dx=1, dy=1, ksize=5) 
        if direction == "HORIZONTAL":
            frame = sobelx
            return self.put_subtitles(frame, "Horizontal edges")
        elif direction == "VERTICAL":
            frame = sobely
            return self.put_subtitles(frame, "Vertical edges")
        else:
            frame = sobelxy
            return self.put_subtitles(frame, "Combined edges")
        
    # Code was fetched from OpenCV and adjusted    
    def hough_transform(self, frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blur = cv.medianBlur(gray, 5)

        rows = gray.shape[0]

        circles = cv.HoughCircles(blur, cv.HOUGH_GRADIENT, 1, rows / 8,
                               param1=90, param2=40,
                               minRadius=250, maxRadius=350)
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                # circle center
                cv.circle(frame, center, 1, (0, 100, 100), 3)
                # circle outline
                radius = i[2]
                cv.circle(frame, center, radius, (255, 0, 255), 3)
        return frame
    
    def detect_ball(self, frame, img):
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        template = cv.imread(img, 0)
        #gray_template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

        w, h = template.shape[::-1]

        res = cv.matchTemplate(gray_frame, template, cv.TM_CCOEFF_NORMED)
        threshold = 0.85
        loc = np.where(res >= threshold)

        for pt in zip(*loc[::-1]):
            cv.rectangle(frame, pt, (pt[0]+w, pt[1]+h), (0,255,255), 2)
        
        return frame





    
    def play(self):
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        output_video = cv.VideoWriter("test.mp4", fourcc, self.get_fps(), (self.get_width(), self.get_height()))
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            if self.between(0, 4000):
                frame = self.detect_ball(frame, 'assets/blue_ball.png')

            '''
            if self.between(0, 1000):
                self.put_subtitles(frame, "GRAY")
                frame = self.switch_color(frame, "GRAY")
                frame = self.switch_color(frame, "BGR")
            
            elif self.between(1000, 2000):
                self.put_subtitles(frame, "BGR")

            elif self.between(2000, 3000):
                self.put_subtitles(frame, "GRAY")
                frame = self.switch_color(frame, "GRAY")
                frame = self.switch_color(frame, "BGR")
            
            elif self.between(3000, 4000):
                self.put_subtitles(frame, "BGR")

            elif self.between(4000, 6000):
                frame = self.gaussian_blur(frame, 5)
            
            elif self.between(6000, 8000):
                frame = self.gaussian_blur(frame, 15)

            elif self.between(8000, 10000):
                frame = self.bilateral_filter(frame, 15)
            
            elif self.between(10000, 12000):
                frame = self.bilateral_filter(frame, 75)
            
            if self.between(0, 4000):
                frame = self.grab_object_rgb(frame)
            elif self.between(4000, 8000):
                frame = self.grab_object_hsv(frame)

                        if self.between(0, 4000):
                frame = self.detect_edges(frame, "HORIZONTAL")

            elif self.between(4000, 8000):
                frame = self.detect_edges(frame, "VERTICAL")
            
            elif self.between(8000, 12000):
                frame = self.detect_edges(frame, "COMBINED")
            if self.between(0, 4000):
                frame = self.hough_transform(frame)
            '''
            
            output_video.write(frame)

            cv.imshow('frame', frame)
            if cv.waitKey(1) == ord('q'):
                break
            cv.waitKey(int(1000 / self.get_fps()))
        self.cap.release()
        output_video.release()
        cv.destroyAllWindows()

    
        


    
video = Video('assets/extended_three_balls.mp4')
video.play()
        
#video = Video2('assets/three_balls.mp4')
#video.repeat_clip(8)