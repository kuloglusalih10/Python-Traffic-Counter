import cv2
import numpy as np
from time import sleep


min_dikdortgen_gen = 65
min_dikdortgen_yuk = 65
maks_dikdortgen_gen = 150
maks_dikdortgen_yuk = 150

offset = 6


y_border = 210
x_border = 320

delay = 60

tespit = []

arac_sayisi = 0
giren_arac_sayisi = 0
cikan_arac_sayisi = 0


def merkez_al(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy


cap = cv2.VideoCapture('video_2.mp4')

backg_sub = cv2.createBackgroundSubtractorKNN()

while True:

    ret, frame = cap.read()
    tmp = float(1 / delay)
    sleep(tmp)

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grey, (21, 21), 0)
    sub_img = backg_sub.apply(blurred)


    dilate = cv2.dilate(sub_img, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Morphologhical operations
    opening = cv2.morphologyEx(dilate, cv2.MORPH_OPEN, kernel, iterations=3)
    eroding = cv2.morphologyEx(opening, cv2.MORPH_ERODE, kernel, iterations=2)
    closing = cv2.morphologyEx(eroding, cv2.MORPH_CLOSE, kernel, iterations=3)


    # contour
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame, (25, y_border), (600, y_border), (255, 255, 0), 2)
    cv2.line(frame, (x_border, 25), (x_border, 340), (0, 0, 0), 2)

    for (i, c) in enumerate(contours):

        (x, y, w, h) = cv2.boundingRect(c)  # (x,y) = dikdörtgenin sol üst koşesinin kordinatları

        contour_kontrol = (w >= min_dikdortgen_gen) and (h >= min_dikdortgen_yuk) and w < 160 and h < 160
        if not contour_kontrol:
            continue

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        merkez = merkez_al(x, y, w, h)
        tespit.append(merkez)
        cv2.circle(frame, merkez, 4, (0, 0, 255), -1)

        for (x, y) in tespit:
            if y < (y_border + offset) and y > (y_border - offset) and x <= x_border:
                giren_arac_sayisi += 1
                arac_sayisi += 1
                cv2.line(frame, (25, y_border), (600, y_border), (0, 0, 255), 2)
                tespit.clear()

            if y < (y_border + offset) and y > (y_border - offset) and x >= x_border:
                cikan_arac_sayisi += 1
                arac_sayisi += 1
                tespit.clear()
                cv2.line(frame, (25, y_border), (600, y_border), (34, 233, 29), 2)

    cv2.putText(frame, "Toplam Arac Sayisi : " + str(arac_sayisi), (300, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9,
                (0, 0, 0), 2)
    cv2.putText(frame, "Giren Arac Sayisi : " + str(giren_arac_sayisi), (300, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9,
                (0, 0, 255), 2)
    cv2.putText(frame, "Cikan Arac Sayisi : " + str(cikan_arac_sayisi), (300, 80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9,
                (34, 233, 29), 2)

    cv2.imshow("Cikti", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()
