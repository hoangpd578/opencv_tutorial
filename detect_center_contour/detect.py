import cv2
import argparse
import imutils
import argparse

def find_center_contour(imagPath):
    image = cv2.imread(imagPath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        M = cv2.moments(c)
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])

        cv2.circle(image, (cX, cY), 4, (255, 0, 0), -1)
        cv2.drawContours(image, [c], -1, (255, 0, 0), 1)
        cv2.putText(image, "Center", (cX-20, cY-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),
                    1)
    cv2.imshow('iamge', image)
    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()
    elif key == ord("s"):
        cv2.imwrite("center.png", image)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--imagePath")
    args = vars(ap.parse_args())
    imagePath = args['imagePath']
    find_center_contour(imagePath)