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
    face.transform_image(tom, vin, tom_triangles, vin_triangles, warped)

    cv.imshow("tom", tom)
    cv.imshow("vin", vin)
    cv.imshow("warped", warped)
    cv.waitKey(0)

    face.draw_landmarks(tom, tom_faces)
    face.draw_landmarks(vin, vin_faces)
    cv.imshow("tom", tom)
    cv.imshow("vin", vin)
    cv.imshow("warped", warped)
    cv.waitKey(0)

    face.draw_triangles(tom, tom_triangles)
    face.draw_triangles(vin, vin_triangles)
    cv.imshow("tom", tom)
    cv.imshow("vin", vin)
    cv.imshow("warped", warped)
    cv.waitKey(0)

    cv.destroyAllWindows()

    mark('picasso.jpg')
    cv.waitKey(0)

    mark('goku.jpg')
    cv.waitKey(0)

    mark('anime.jpg')
    cv.waitKey(0)

    mark('anime2.jpg')
    cv.waitKey(0)

    mark('bart.jpg')
    cv.waitKey(0)

    mark('trump.png')
    cv.waitKey(0)

    cv.destroyAllWindows()


def mark(filename):
    im = cv.imread(filename)
    faces = face.get_landmarks(im)
    face.draw_landmarks(im, faces)

    cv.imshow(filename, im)


if __name__ == "__main__":
    main()
