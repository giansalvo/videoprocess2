"""
    Application of Neural Network for the segmentation of Substantia Nigra
    in Ultrasound videos for the detection of Parkinson's Disease.

    Copyright (c) 2022 Giansalvo Gusinu

    Permission is hereby granted, free of charge, to any person obtaining a 
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
"""
import os
import logging
import argparse
import datetime

import tensorflow as tf

import numpy as np
import cv2
import matplotlib.pyplot as plt # TODO DEBUG ONLY

SAVE_PATH = "videos"
WINDOW_NAME = "Preview windows (ESC to quit)"
MODEL_PATH = "model_unet_us_w46.h5"

# COPYRIGHT NOTICE AND PROGRAM VERSION
COPYRIGHT_NOTICE = "Copyright (C) 2022 Giansalvo Gusinu"
PROGRAM_VERSION = "1.0"


# TODO DEBUG ONLY
def read_image(image_path, channels=3):
    img_size = 256 # TODO HARDCODED
    img0 = tf.io.read_file(image_path)
    img0 = tf.image.decode_jpeg(img0, channels=channels)
    height, width, _ = img0.shape
    img0 = tf.image.resize(img0, [img_size,img_size])
    return img0, width, height


# TODO DEBUG ONLY
def plot_samples_matplotlib(display_list, labels_list=None, figsize=None, fname=None):
    NIMG_PER_COLS = 6
    if figsize is None:
        PX = 1/plt.rcParams['figure.dpi']  # pixel in inches
        figsize = (600*PX, 300*PX)
    ntot = len(display_list)
    if ntot <= NIMG_PER_COLS:
        nrows = 1
        ncols = ntot
    elif ntot % NIMG_PER_COLS == 0:
        nrows = ntot // NIMG_PER_COLS
        ncols = NIMG_PER_COLS
    else:
        nrows = ntot // NIMG_PER_COLS + 1
        ncols = NIMG_PER_COLS
    _, axes = plt.subplots(nrows, ncols, figsize=figsize)
    for i_img in range(ntot):
        i = i_img // NIMG_PER_COLS
        j = i_img % NIMG_PER_COLS
        if display_list[i_img].shape[-1] == 3:
                img = tf.keras.preprocessing.image.array_to_img(display_list[i_img])
                if nrows > 1:
                    if labels_list is not None:
                        axes[i, j].set_title(labels_list[i_img])
                    axes[i, j].imshow(img, cmap='Greys_r')
                else:
                    if labels_list is not None:
                        axes[i_img].set_title(labels_list[i_img])
                    axes[i_img].imshow(img, cmap='Greys_r')
        else:
                img = display_list[i_img]
                if nrows > 1:
                    if labels_list is not None:
                        axes[i, j].set_title(labels_list[i_img])
                    axes[i, j].imshow(img, cmap='Greys_r')
                else:
                    if labels_list is not None:
                        axes[i_img].set_title(labels_list[i_img])
                    axes[i_img].imshow(img, cmap='Greys_r')
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for axes in axes.flat:
        axes.label_outer()

    if fname is None:
        plt.show()
    else:
        #logger.debug("Saving prediction to file {}...".format(fname))
        plt.savefig(fname)
        plt.close()


def create_mask(pred_mask: tf.Tensor) -> tf.Tensor:
    """Return a filter mask with the top 1 predicitons
    only.

    Parameters
    ----------
    pred_mask : tf.Tensor
        A [img_size, img_size, N_CLASS] tensor. For each pixel we have
        N_CLASS values (vector) which represents the probability of the pixel
        being these classes. Example: A pixel with the vector [0.0, 0.0, 1.0]
        has been predicted class 2 with a probability of 100%.

    Returns
    -------
    tf.Tensor
        A [img_size, img_size, 1] mask with top 1 predictions
        for each pixels.
    """
    # pred_mask -> [img_size, img_size, N_CLASS]
    # 1 prediction for each class but we want the highest score only
    # so we use argmax
    pred_mask = tf.argmax(pred_mask, axis=-1)
    # pred_mask becomes [img_size, img_size]
    # but matplotlib needs [img_size, img_size, 1]
    pred_mask = tf.expand_dims(pred_mask, axis=-1)
    return pred_mask


def image_fusion(background, foreground, sharp=False, alfa=0.5):
    img1 = background
    img2 = foreground

    rows,cols,channels = img2.shape
    roi = img1[0:rows, 0:cols ]

    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 200, 255, cv2.THRESH_BINARY_INV)
    mask_inv = cv2.bitwise_not(mask)

    img1_hidden_bg = cv2.bitwise_and(img1, img1, mask = mask)
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

def get_overlay(img_sample, img_pred, img_gt=None):
    # overlay between sample image, ground truth and prediction
    FOREGROUND = 1
    OFFSET = 255 - FOREGROUND
    ALFA = 0.5
    BETA = 1 - ALFA
    RED = [0,0, 255] # BGR
    WHITE = [255,255,255]
    COLOR_WHITE_LOWER = np.array([0, 0, 255])
    COLOR_WHITE_UPPER = np.array([180, 255, 255])
    CONTOUR_COLOR = (0, 255, 0)  # green contour (BGR)
    CONTOUR_THICK = 2

    #####
    # fusion of sample image and foreground area from predicted image
    #####
    img_pred += OFFSET
    img_pred=cv2.cvtColor(img_pred, cv2.COLOR_GRAY2BGR)
    img_pred[np.all(img_pred == WHITE, axis=-1)] = RED

    # result = cv2.addWeighted( img_sample, ALFA, img_pred, BETA, 0.0)
    result = cv2.addWeighted( img_sample, ALFA, img_pred, BETA, 0.0)

    # plot_samples_matplotlib([img_sample, img_pred, result], ["sample", "prediction", "weighted result"] )

    #####
    # extract contour of foreground area from ground truth
    #####
    if img_gt is not None:
        img_gt += OFFSET
        img_gt=cv2.cvtColor(img_gt, cv2.COLOR_GRAY2BGR)
        # change color space and set color mask
        imghsv = cv2.cvtColor(img_gt, cv2.COLOR_BGR2HSV)
        mask_color = cv2.inRange(imghsv, COLOR_WHITE_LOWER, COLOR_WHITE_UPPER)
        # get close contour
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(3, 3))
        img_close_contours = cv2.morphologyEx(mask_color, cv2.MORPH_CLOSE, kernel, iterations=1)
        # Find outer contours
        cnts, _ = cv2.findContours(img_close_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # img_contours = np.zeros((img_gt.shape[0], img_gt.shape[1], 3), dtype="uint8")  # RGB image black
        # and draw on previously prepared image
        cv2.drawContours(result, cnts, -1, CONTOUR_COLOR, CONTOUR_THICK)

    return result


def park_detection2(model, img):
    global logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('gians')

    w_orig = img.shape[1]
    h_orig = img.shape[0]

    # img0, _, _ = read_image("I000.jpg") 
    img0 = tf.image.resize(img, [256, 256]) # TODO HARDCODED VALUEs
    img_tensor = tf.cast(img0, tf.float32) / 255.0    # normalize
    img_tensor = np.expand_dims(img_tensor, axis=0)

    predictions = model.predict(img_tensor)
    pred = create_mask(predictions)[0]
    pred = pred + 1 # de-normalization

    # plot_samples_matplotlib([img0, pred], ["sample", "prediction"] )

    # convert to OpenCV image format
    i0 = img0.numpy()
    i1 = pred.numpy()
    i1 = np.squeeze(i1)
    i1 = np.float32(i1)

    # overlay = get_overlay(i0, i1)
    
    FOREGROUND = 1
    OFFSET = 255 - FOREGROUND
    RED = [0,0, 255] # BGR
    WHITE = [255,255,255]

    #####
    # fusion of sample image and foreground area from predicted image
    #####
    img_pred = i1
    img_pred += OFFSET
    img_pred=cv2.cvtColor(img_pred, cv2.COLOR_GRAY2BGR)
    img_pred[np.all(img_pred == WHITE, axis=-1)] = RED
    i1 = img_pred

    i0 = i0.astype(np.uint8)
    i1 = i1.astype(np.uint8)
   
    overlay = image_fusion(i0, i1)

    overlay = cv2.resize(overlay, (w_orig, h_orig))

    # cv2.imwrite("pippo.png", overlay,  [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

    return overlay

def main():
    # create logger
    logger = logging.getLogger('gians')
    logger.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s:%(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)
    logger.info("Starting")

    parser = argparse.ArgumentParser(
        description=COPYRIGHT_NOTICE,
        epilog = "Examples:\n"
                "       Analyse the input video (.mp4 format) and save output video.\n"
                "         $python %(prog)s -i input_video\n"
                "\n",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--version', action='version', version='%(prog)s v.' + PROGRAM_VERSION)
    #group = parser.add_mutually_exclusive_group()
    #group.add_argument("-v", "--verbose", action="store_true")
    #group.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument('-i', '--input_video', required=True, help="The video to be analysed (.mp4 format).")
    args = parser.parse_args()

    video_input = args.input_video;

    print("Loading network model from: " + MODEL_PATH)
    # model = tf.keras.models.load_model(network_structure_path, custom_objects={'dice_coef': dice_coef})
    model = tf.keras.models.load_model(MODEL_PATH)

    cv2.namedWindow(WINDOW_NAME)
    vc = cv2.VideoCapture(video_input)
    vid_cod = cv2.VideoWriter_fourcc(*'mp4v')
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    OUTPUT_PATH = os.path.join(SAVE_PATH, "video_" + timestamp + ".mp4")
    print("Saving analysed video to: " + OUTPUT_PATH)
    
    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
        h, w, c = frame.shape
        print("Video resolution: {} x {} x {}".format(w, h, c))
        output = cv2.VideoWriter(OUTPUT_PATH, vid_cod, 20.0, (w, h))
    else:
        print("Error opening the stream.")
        rval = False

    cv2.imshow(WINDOW_NAME, frame)

    while rval:

        frame = park_detection2(model, frame)

        cv2.imshow(WINDOW_NAME, frame)
        output.write(frame)
        rval, frame = vc.read()

        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break

    cv2.destroyWindow(WINDOW_NAME)
    vc.release()
    print("Program terminated.")
    return

if __name__ == '__main__':
    main()
