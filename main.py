import cv2 as cv
import numpy as np
import face


def main():
    tom = cv.imread('tom.jpg')
    tom_faces = face.get_landmarks(tom)
    tom_triangles = face.triangulate(tom, tom_faces[0])

    vin = cv.imread('vin.jpg')
    vin_faces = face.get_landmarks(vin)
    vin_triangles = face.triangulate(vin, vin_faces[0])

    warped = 255 * np.ones(vin.shape, vin.dtype)
    # face.transform_image(tom, vin, tom_triangles, vin_triangles, warped)

    cv.imshow("tom", tom)
    cv.imshow("vin", vin)

    mark('picasso.jpg')

    # mark('goku.jpg')

    # mark('anime.jpg')

    cv.waitKey(0) & 0xFF
    cv.destroyAllWindows()


def mark(filename):
    im = cv.imread(filename)
    faces = face.get_landmarks(im)
    face.draw_landmarks(im, faces)

    cv.imshow(filename, im)


if __name__ == "__main__":
    main()
