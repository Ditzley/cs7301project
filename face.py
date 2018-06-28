import numpy as np
import cv2 as cv
import dlib


def get_landmarks(mat):
    # cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv.cvtColor(mat, cv.COLOR_BGR2GRAY)
    #
    # faces = cascade.detectMultiScale(gray, 1.3, 5)
    #
    # for(x, y, w, h) in faces:
    #     cv.rectangle(mat, (x, y), (x + w, y + h), (255, 0, 0), 2)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    faces = detector(gray, 1)

    out = list()

    for (f, face) in enumerate(faces):
        shape = predictor(gray, face)

        (x, y, w, h) = (face.left(), face.top(), face.right() - face.left(), face.bottom() - face.top())
        # cv.rectangle(mat, (x, y), (x + w, y + h), (255, 0, 0), 2)

        coords = np.zeros((68, 2), "int")
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)

        out.append({'face': (x, y, w, h), 'landmarks': coords})

    return out


def draw_landmarks(mat, faces):
    for (f, face) in enumerate(faces):
        print(faces[f])
        for (x, y) in faces[f].get('landmarks'):
            cv.circle(mat, (x, y), 2, (0, 0, 255), -1)


def triangulate(mat, face):
    size = mat.shape
    rect = (0, 0, size[1], size[0])

    subdiv = cv.Subdiv2D(rect)
    for (c, coords) in enumerate(face.get('landmarks')):
        (x, y) = coords
        subdiv.insert((x, y))

    triangles = subdiv.getTriangleList()
    points = list()
    for t in triangles:
        p1 = (t[0], t[1])
        p2 = (t[2], t[3])
        p3 = (t[4], t[5])

        if point_in_rect(rect, p1) and point_in_rect(rect, p2) and point_in_rect(rect, p3):
            # cv.line(mat, p1, p2, (0, 255, 0), 1, 0)
            # cv.line(mat, p2, p3, (0, 255, 0), 1, 0)
            # cv.line(mat, p3, p1, (0, 255, 0), 1, 0)
            points.append(np.array([p1, p2, p3]))

    return points


def transform_image(mat1, mat2, t1, t2, out):
    # out = mat1
    # out = cv.m
    # for(i, s) in enumerate(source):
    #     t = target[i]
    #     transform = cv.getAffineTransform(s, t)
    #     # print(s)
    #     dst = cv.warpAffine(s, transform, )
    #     break
    # cv.imshow('m', mat1)
    for (i, t) in enumerate(t1):
        transform_triangle(mat1, t, t2[i], out)


def transform_triangle(mat, t1, t2, out):
    r1 = cv.boundingRect(t1)
    r2 = cv.boundingRect(t2)

    t1_cropped = []
    t2_cropped = []

    for i in range(0, 3):
        t1_cropped.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2_cropped.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    s = mat[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

    transform = cv.getAffineTransform(np.float32(t1_cropped), np.float32(t2_cropped))
    t = cv.warpAffine(s, transform, (r2[2], r2[3]), None, flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT_101)

    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv.fillConvexPoly(mask, np.int32(t2_cropped), (1.0, 1.0, 1.0), 16, 0)

    t[mask == 0] = 0

    out[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = out[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (
                (1.0, 1.0, 1.0) - mask)
    out[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = out[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + t

    # out[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = t
    # print('t')
    # print(t)
    # print('out')
    # print(out[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]])

    cv.imshow('image', out)
    cv.waitKey(0) & 0xFF
    return


def point_in_rect(rect, point):
    return rect[0] < point[0] < rect[0] + rect[2] and rect[1] < point[1] < rect[1] + rect[3]
