from dmtools.image import *


class BaseImage(object):

    def __init__(self):
        pass

    def get_origin_image(self, path):
        """
        加载原始图片
        """
        return np.copy(cv2.imread(path))

    def show_image(self, image):
        """
        显示图片
        """
        plt.figure(figsize=(18,15))
        #Before showing image, bgr color order transformed to rgb order
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def detect_face(self, image, scaleFactor, minNeighbors, minSize):
        # face will detected in gray image
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(image_gray,
                                              scaleFactor=scaleFactor,
                                              minNeighbors=minNeighbors,
                                              minSize=minSize)

        for x, y, w, h in faces:
            # detected faces shown in color image
            cv2.rectangle(image, (x, y), (x + w, y + h), (127, 255, 0), 3)

        self.show_image(image)