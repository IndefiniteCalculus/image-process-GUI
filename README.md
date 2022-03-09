# Image-process-GUI
This project is a final project of "Digital Image Processing". It aims to process the image in an easy and simple way. The options include many classic digital image processing methods such as grayscale, thresholding, histogram equalization, various filters and so on. Each stage of image during the processing can be displayed as well.The user can also undo the previous operation. The project might be helpful for some teaching purpose, to show what happened to the image during the processing.

# Requires
The code should be compiled by python 3.7, make sure you had installed opencv-python and  tkinter to run the code.

# How to use it
The project can be used to process the image with various image processing methods by select options in menubar. 

You can see the procedures in text box on the right side. And a botton named "查看" on the buttom of text box. Select one of these procedures in the text box, then click the botton below, and you can see what your picture looks like after your selected procedure finished. 

You can back to specific stage of image after several image processing procedures by select "返回上一张图像" option in menu "文件", if you are not satisfied with the processing result.

You can load the image by click "打开" in the menu "文件", then select image file in your computer, just make sure you had change the select mode into "所有文件" if your image is not saved as .PNG format.

The image processing methods supported so far has: 
rotate, scale, gray(if your image is RGB image), binary, threshold, some noise remove and sharpening filters, some image inhance methods, morphological transform like dilate, erode, open and close operation(with adjustable parameters).

# Unsolved bug
There is a bug I can't solved here, sometimes the program might stuck and exit when I try to open some files, I can't figure out why. I anticipate it might cause by one of the components of tkinter, but I don't know how tkinter implement it yet so I can't comfirm my doubts.
