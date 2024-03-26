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
        return cv.resize(frame, (1920, 1080))

    def get_current_time(self):
        return int(self.cap.get(cv.CAP_PROP_POS_MSEC))
    
    def between(self, begin, end):
        return begin <= self.get_current_time() < end

    def get_fps(self):
        return int(self.cap.get(5))
    
    def put_subtitles(self, frame, text, type):
        if type == "label":
            return cv.putText(frame, text, (50,50), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        else:
            return cv.putText(frame, text, (10, 1000), cv.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)

    def switch_color(self, frame, new_color):
        colors = {
            "GRAY": cv.COLOR_BGR2GRAY,
            "BGR": cv.COLOR_GRAY2BGR
        }
        return cv.cvtColor(frame, colors[new_color])
    
    def gaussian_blur(self, frame, kernel):
        frame = cv.GaussianBlur(frame, (kernel, kernel), 0)
        self.put_subtitles(frame, f"GAUSSIAN with kernel of size {kernel}", "label")
        return frame
    
    def bilateral_filter(self, frame, kernel):
        frame = cv.bilateralFilter(frame, 15, kernel, kernel)
        self.put_subtitles(frame, f"BILATERAL with kernel of size {kernel}", "label")
        return frame
    
    def grab_object_rgb(self, frame):
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        lower_blue = np.array([0, 0, 100])    
        upper_blue = np.array([100, 100, 255]) 

        mask = cv.inRange(rgb, lower_blue, upper_blue)
        return cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    
    def grab_object_hsv(self, frame):
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        lower_blue = np.array([100, 50, 50])   
        upper_blue = np.array([130, 255, 255]) 

        mask = cv.inRange(hsv, lower_blue, upper_blue)
        return cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    
    def improve_grabbing(self, frame):
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])

        
        mask = cv.inRange(hsv, lower_blue, upper_blue)
        
        kernel = np.ones((6, 6), np.uint8)
        eroded_mask = cv.erode(mask, kernel, iterations=2)
        improved_mask = cv.morphologyEx(eroded_mask, cv.MORPH_OPEN, kernel)
        
        inverted_mask = cv.cvtColor(improved_mask, cv.COLOR_GRAY2BGR)
        inverted_mask[np.where((inverted_mask == [0, 255, 0]).all(axis=2))] = [255, 255, 255]  
        inverted_mask[np.where((inverted_mask == [255, 255, 255]).all(axis=2))] = [0, 255, 0]  
        
        return inverted_mask

    
    def detect_edges(self, frame, direction):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (3,3), 0)
        sobelx = cv.Sobel(src=blur, ddepth=cv.CV_64F, dx=1, dy=0, ksize=5) 
        sobely = cv.Sobel(src=blur, ddepth=cv.CV_64F, dx=0, dy=1, ksize=5) 
        sobelxy = cv.Sobel(src=blur, ddepth=cv.CV_64F, dx=1, dy=1, ksize=5) 
        
        if direction == "HORIZONTAL":
            edges = sobelx
            subtitle_text = "Horizontal edges"
        elif direction == "VERTICAL":
            edges = sobely
            subtitle_text = "Vertical edges"
        else:
            edges = sobelxy
            subtitle_text = "Combined edges"
        
        edges = cv.convertScaleAbs(edges)
        
        edges_bgr = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
        
        return self.put_subtitles(edges_bgr, subtitle_text, "label")

        
    # Code was fetched from OpenCV and adjusted    
    def hough_transform(self, frame, dp, min_dist, param1, param2, min_radius, max_radius):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blur = cv.medianBlur(gray, 5)

        circles = cv.HoughCircles(blur, cv.HOUGH_GRADIENT, dp, min_dist,
                                  param1=param1, param2=param2,
                                  minRadius=min_radius, maxRadius=max_radius)
        
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
        threshold = 0.7
        loc = np.where(res >= threshold)

        for pt in zip(*loc[::-1]):
            cv.rectangle(frame, pt, (pt[0]+w, pt[1]+h), (0,255,255), 2)
        
        return frame
    
    def likelihood(self, frame, img):
        template = cv.imread(img, 0)

        res = 1 - cv.matchTemplate(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), template, cv.TM_CCOEFF_NORMED)

        res_norm = cv.normalize(res, None, 0, 255, cv.NORM_MINMAX)

        res_uint8 = res_norm.astype('uint8')

        res_bgr = cv.cvtColor(res_uint8, cv.COLOR_GRAY2BGR)

        frame_height, frame_width = frame.shape[:2]
        res_height, res_width = res_bgr.shape[:2]

        top = (frame_height - res_height) // 2
        bottom = frame_height - res_height - top
        left = (frame_width - res_width) // 2
        right = frame_width - res_width - left

        res_bgr_padded = cv.copyMakeBorder(res_bgr, top, bottom, left, right, cv.BORDER_CONSTANT, value=(0, 0, 0))

        return res_bgr_padded
    
    def cartoonify(self, frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = cv.medianBlur(gray, 5)
        outlines = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 9, 9)

        color = cv.bilateralFilter(frame, 9, 250, 250)
        cartoon = cv.bitwise_and(color, color, mask=outlines)

        return cartoon
    
    def sift_textured_ball(self, frame):
        textured_ball = cv.imread('assets/textured_ball.png', cv.IMREAD_GRAYSCALE)

        
        sift = cv.SIFT_create()
        kp1, des1 = sift.detectAndCompute(textured_ball, None)
        kp2, des2 = sift.detectAndCompute(frame, None)

        bf = cv.BFMatcher()

        matches = bf.knnMatch(des1, des2, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        matched_frame = cv.drawMatches(textured_ball, kp1, frame, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        matched_frame = self.resize(matched_frame)
        return matched_frame
    
    # https://medium.com/dataseries/designing-image-filters-using-opencv-like-abode-photoshop-express-part-2-4479f99fb35
    def sepia_filter(self, frame):
        frame = cv.transform(frame, np.matrix([[0.272, 0.534, 0.131],
                                    [0.349, 0.686, 0.168],
                                    [0.393, 0.769, 0.189]]))
        frame[np.where(frame > 255)] = 255
        return frame
    
    # https://stackoverflow.com/questions/55508615/how-to-pixelate-image-using-opencv-in-python
    def pixelate_filter(self, frame):
        height, width = frame.shape[:2]
        w, h = (64, 64)
        frame = cv.resize(frame, (w,h), interpolation=cv.INTER_LINEAR)
        frame = cv.resize(frame, (width, height), interpolation=cv.INTER_NEAREST)
        return frame
    
    def pencil_sketch(self, frame):
        gray_sketch = cv.pencilSketch(frame, sigma_s=20, sigma_r=0.5, shade_factor=0.02)
        return gray_sketch
    
    def sharpening_effect(self, frame):
        kernel = np.array([[-1, -1, -1],
                           [-1, 9.2, -1],
                           [-1, -1, -1]])
        return cv.filter2D(src=frame, ddepth=-1, kernel=kernel)

    def play(self):
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        output_video = cv.VideoWriter("test.mp4", fourcc, self.get_fps(), (self.get_width(), self.get_height()))
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            if self.between(0, 1000):
                frame = self.switch_color(frame, "GRAY")
                frame = self.switch_color(frame, "BGR")
                self.put_subtitles(frame, "GRAY", "label")
            
            elif self.between(1000, 2000):
                self.put_subtitles(frame, "BGR", "label")

            elif self.between(2000, 3000):
                frame = self.switch_color(frame, "GRAY")
                frame = self.switch_color(frame, "BGR")
                self.put_subtitles(frame, "GRAY", "label")
            
            elif self.between(3000, 4000):
                self.put_subtitles(frame, "BGR", "label")

            elif self.between(4000, 6000):
                frame = self.gaussian_blur(frame, 5)
                self.put_subtitles(frame, "The Gaussian filter is an example of a linear filter, it replaces each pixel by a linear combination of its neighbors.", "sub")

            elif self.between(6000, 8000):
                frame = self.gaussian_blur(frame, 15)
                self.put_subtitles(frame, "This obviously means that neighbouring pixels have a higher influence.", "sub")

            elif self.between(8000, 10000):
                frame = self.bilateral_filter(frame, 15)
                self.put_subtitles(frame, "The bilateral filter is an example of a non-linear filter, it considers both the spatial distance and intensity differences between pixels.", "sub")

            elif self.between(10000, 12000):
                frame = self.bilateral_filter(frame, 75)
                self.put_subtitles(frame, "The bilateral filter notably preserves edges better than the Gaussian.", "sub")
            
            elif self.between(12000, 14000):
                frame = self.grab_object_rgb(frame)
                self.put_subtitles(frame, "Grab in RGB", "label")

            elif self.between(14000, 16000):
                frame = self.grab_object_hsv(frame)
                self.put_subtitles(frame, "Grab in HSV", "label")

            elif self.between(16000, 19000):
                frame = self.improve_grabbing(frame)
                self.put_subtitles(frame, "Improved grabbing using erosion and morph. operations", "label")

            elif self.between(19000, 20000):
                frame = self.grab_object_hsv(frame)
                self.put_subtitles(frame, "Grab in HSV", "label")
            
            
            if self.between(20000, 21500):
                frame = self.detect_edges(frame, "HORIZONTAL")

            elif self.between(21500, 23000):
                frame = self.detect_edges(frame, "VERTICAL")
            
            elif self.between(23000, 25000):
                frame = self.detect_edges(frame, "COMBINED")

            elif self.between(25000, 28000):
                frame = self.hough_transform(frame, dp=1, min_dist=frame.shape[0],
                                                  param1=40, param2=20,
                                                  min_radius=20, max_radius=150)
                self.put_subtitles(frame, "Lower min radius and thresholds", "label")

            elif self.between(28000, 31000):
                frame = self.hough_transform(frame, dp=1, min_dist=frame.shape[0] / 8,
                                                  param1=90, param2=40,
                                                  min_radius=20, max_radius=150)
                self.put_subtitles(frame, "Higher thresholds and lower min radius", "label")
            
            elif self.between(31000, 35000):
                frame = self.hough_transform(frame, dp=1, min_dist=frame.shape[0] / 8,
                                                  param1=90, param2=40,
                                                  min_radius=250, max_radius=350)
                self.put_subtitles(frame, "Higher min radius and thresholds (best params)", "label")

            elif self.between(35000, 38000):
                frame = self.detect_ball(frame, 'assets/blue_ball.png')
                self.put_subtitles(frame, "Object detection through Hough", "label")
            
            elif self.between(38000, 40000):
                frame = self.likelihood(frame, 'assets/blue_ball.png')
                self.put_subtitles(frame, "Likelihood map", "label")
                self.put_subtitles(frame, "Uses the matchTemplate which returns N-n+1 x N-n+1 of dissimilarity values, take the inverse of this number and normalize so 0 = 0 and 1 = 250", "sub")
            
            elif self.between(40000, 45000):
                frame = self.cartoonify(frame)
                self.put_subtitles(frame, "Cartoon effect", "label")
            
            elif self.between(45000, 50000):
                frame = self.sepia_filter(frame)
                self.put_subtitles(frame, "Sepia effect", "label")
            
            elif self.between(50000, 53000):
                frame = self.pixelate_filter(frame)
                self.put_subtitles(frame, "Pixelate effect", "label")
            
            elif self.between(53000, 55000):
                frame = self.sharpening_effect(frame)
                self.put_subtitles(frame, "Sharpening effect", "label")

            elif self.between(55000, 60000):
                frame = self.sift_textured_ball(frame)
                self.put_subtitles(frame, "Image matching with SIFT", "label")
                self.put_subtitles(frame, "SIFT is scale and rotation invariant, meaning that it can detect keypoints across different scales and rotations.", "sub")
            
            elif self.between(60000, float('inf')):
                break
            
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