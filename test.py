import cv2


def image_fusion(background, foreground, sharp=False, alfa=0.5):
    img1 = background
    img2 = foreground

    rows,cols,channels = img2.shape
    roi = img1[0:rows, 0:cols ]

    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 200, 255, cv2.THRESH_BINARY_INV)
    mask_inv = cv2.bitwise_not(mask)
    #cv2.imshow("mask", mask)
    #cv2.imshow("mask inv", mask_inv)

    img1_hidden_bg = cv2.bitwise_and(img1,img1,mask = mask)
    #cv2.imshow("img1_hidden_bg", img1_hidden_bg)

    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
    #cv2.imshow("img1_bg", img1_bg)

    img2_fg = cv2.bitwise_and(img2,img2,mask = mask)
    #cv2.imshow("img2_fg", img2_fg)

    out_img = cv2.add(img1_bg,img2_fg)
    #cv2.imshow("out_img", out_img)
    img1[0:rows, 0:cols ] = out_img

    # cv2.imshow("Result Sharp", img1)

    ALFA = 0.5
    img1_hidden_merge = cv2.addWeighted(img1_hidden_bg, ALFA, img2_fg, 1-ALFA, 0.0)
    img_opaque = cv2.add(img1_bg, img1_hidden_merge)

    if sharp:
        return img1
    else:
        return img_opaque


img1 = cv2.imread('I000.jpg')
img2 = cv2.imread('image_SN.jpg')

img = image_fusion(img1, img2, alfa=0.8)
cv2.imshow("Result Opaque", img)

img = image_fusion(img1, img2, sharp=True)
cv2.imshow("Result Sharp", img)

cv2.waitKey(0)

cv2.destroyAllWindows()
print("Program terminated.")
