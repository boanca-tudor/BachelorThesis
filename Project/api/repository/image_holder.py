class ImageHolder:
    def __init__(self):
        self.__image = None
        self.__raw_image = None

    @property
    def image(self):
        return self.__image

    @image.setter
    def image(self, value):
        self.__image = value

    @property
    def raw_image(self):
        return self.__raw_image

    @raw_image.setter
    def raw_image(self, value):
        self.__raw_image = value