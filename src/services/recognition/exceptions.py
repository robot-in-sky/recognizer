class RecognitionError(Exception):
    ...


class ImageMismatchError(RecognitionError):
    ...


class CornerDetectionError(RecognitionError):
    ...


class TextRecognitionError(RecognitionError):
    ...
