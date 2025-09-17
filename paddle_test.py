from paddleocr import PaddleOCR
ocr = PaddleOCR(
    use_angle_cls=False,
    lang='japan',
    det=True,  # only run text detection
    rec=False, # skip recognition
    cls=False
)

result = ocr.ocr('your_test_image.png', cls=False)
print(result)
