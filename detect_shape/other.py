import cv2
import argparse
import numpy as np
import imutils

def detect_shape(imagePath):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 70, 255, cv2.THRESH_BINARY)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for i in cnts:
        peri = cv2.arcLength(i, True)
        approx = cv2.approxPolyDP(i, 0.04*peri, True)

        if len(approx) == 2:
            shape = "Line"
        elif len(approx) == 3:
            shape = "Triangle"
        elif len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            if 0.95 <= ar <= 1.05:
                shape = "Square"
            else:
                shape = "Rectangle"
        elif  len(approx) == 5:
            shape = "pentagon"
        else:
            shape = "cricle"

        M = cv2.moments(i)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        cv2.drawContours(image, i, -1, (255, 0, 0), 1)
        cv2.putText(image, shape, (cX+10, cY+10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 0), 1)

    cv2.imshow("image", image)
    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()
    elif key == ord('s'):
        cv2.imwrite("detect_shape.jpg", image)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-p', '--pathimage')

    args = vars(ap.parse_args())

    pathimage = args['pathimage']
    detect_shape(pathimage)


