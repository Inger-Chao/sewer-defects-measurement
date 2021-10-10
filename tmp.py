import cv2
from utils.cv_util import empty, stackImages

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters",640,240)
# cv2.namedWindow("APC")
# cv2.resizeWindow("APC", 640, 240)
cv2.createTrackbar("Threshold1","Parameters", 34,255, empty)
cv2.createTrackbar("Threshold2","Parameters", 32,255, empty)
cv2.createTrackbar("Coefficient", "Parameters", 2, 20, empty)
cv2.createTrackbar("thresh", "Parameters", 173, 255, empty)
cv2.createTrackbar("Area","Parameters", 300 ,800, empty)

img = cv2.imread('datasets/level-sewer10/01-tmp/010028.png')
blured = cv2.GaussianBlur(img, (5, 5), 1)
grayed = cv2.cvtColor(blured, cv2.COLOR_BGR2GRAY)

while True:
    canny_threshold1 = cv2.getTrackbarPos("Threshold1","Parameters")
    canny_threshold2 = cv2.getTrackbarPos("Threshold2","Parameters")
    thresh = cv2.getTrackbarPos("thresh", "Parameters")
    imgCanny = cv2.Canny(grayed, canny_threshold1, canny_threshold2)

    # sobel
    grad_x = cv2.Sobel(grayed, -1, 1, 0, ksize=3)
    grad_y = cv2.Sobel(grayed, -1, 0, 1, ksize=3)
    grad  = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)
    # thresh1 is the best, thresh155
    ret,thresh1 = cv2.threshold(src=grayed,thresh=thresh,maxval=255, type=cv2.THRESH_BINARY)
    ret,thresh2 = cv2.threshold(src=grayed,thresh=thresh,maxval=255, type=cv2.THRESH_BINARY_INV)
    ret,thresh3 = cv2.threshold(src=grayed,thresh=thresh,maxval=255,type=cv2.THRESH_TRUNC)
    ret,thresh4 = cv2.threshold(src=grayed,thresh=thresh,maxval=255,type=cv2.THRESH_TOZERO)
    ret,thresh5 = cv2.threshold(src=grayed,thresh=thresh,maxval=255,type=cv2.THRESH_TOZERO_INV)

    coefficient = cv2.getTrackbarPos("Coefficient", "Parameters")
    min_area = cv2.getTrackbarPos("Area","Parameters")
    contours, _ = cv2.findContours(thresh1, cv2.RETR_LIST , cv2.CHAIN_APPROX_NONE)
    img_copy = img.copy()
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            cv2.drawContours(img_copy, cnt, -1, (255, 0, 255), 1)
    imgStack = stackImages(0.4, ([img, imgCanny],
                                [thresh1, img_copy],
                                [thresh2, thresh3],
                                [thresh4, thresh5]))
    cv2.imshow("Parameters", imgStack)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()