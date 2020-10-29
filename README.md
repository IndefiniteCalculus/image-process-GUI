# Image-process-GUI
This project is the course project of "Digital Image Processing". I fixed some bugs, upload the source code and it's exe file here for backup. And might be helpful for some teaching purpose. Free access for everyone.

# Requires
The code should be compiled by python 3.7, make sure you had installed python3 compiler, package cv2 and package tkinter to run the code.

# Introduce
The project can be used to process the image with various image processing methods by select options in menubar. 
You can see the procedures in text box on the right side. And a botton named "查看" on the buttom of text box. Select one of procedures in the text box, click the botton, then you can see what your picture looks like after your selected procedure finished. 
You can back to certain image status after several image processing procedures by select "返回上一张图像" option in menu "文件", you know, if you are not satisfied with the processing result.
You can load the image by click "打开" in the menu "文件", then select image file in your computer, just make sure you had change the select mode into "所有文件" if your image is not saved as .PNG format

The image processing methods supported so far has: rotate, scale, gray(if your image is RGB image), binary, threshold, some noise remove filters, some image inhance methods, morphological transform(with adjustable parameters).

# Unsolved bug
There bug I can't solved here is, in some accidental time, the program might stuck and exit, I can't figure out why. I anticipate it might cause by one of component tkinter, but I hadn't read the source code of tkinter yet to comfirm my doubts.
