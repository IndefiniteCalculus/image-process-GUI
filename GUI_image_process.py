from math import *
from tkinter import *
import cv2 as cv
from PIL import Image,ImageTk
import numpy as np
import tkinter.filedialog as filedialog
import tkinter.simpledialog as simpledialog
class window():
    root:Tk
    # 图像处理时初始图片
    im_src:np.array = None
    # 存储处理后的图像
    im_dist:list = []
    # 仅供显示的图像，会将图像改变到合适的大小
    im_show:np.array = None
    im_tk:PhotoImage
    menubar:Menu
    imageforshow:Canvas
    file_path_open:str
    width:int
    height:int

    def __init__(self):
        self.im_src = cv.imread("./src/Altair_dagger.jpg", cv.COLOR_BGR2GRAY)
        self.initial()
        self.root.mainloop()
# 初始化
    def initial(self):
        self.root = Tk()
        self.root.title("2017210906江笑语 数字图像处理大作业")
        self.width_max,self.height_max=self.root.maxsize()
        self.root.geometry("%sx%s"%(int(self.width_max/2),int(self.height_max/2)))
        self.initial_menu()
        # self.imLable.pack()
        self.imageforshow = Canvas(self.root)
        self.imageforshow.pack()
        self.root.config(menu=self.menubar)
# 初始化菜单按钮，并绑定处理点击时间的函数
    def initial_menu(self):
        self.menubar = Menu(self.root)
        #创建文件子菜单
        self.menu_file=Menu(self.menubar)
        self.menu_file.add_command(label="打开",command=self.openDir)
        self.menu_file.add_command(label="保存",command=self.saveDir)
        self.menu_file.add_command(label="返回上一个图像",command=self.back)
        self.menubar.add_cascade(label="文件",menu=self.menu_file)
        #创建旋转与缩放子菜单
        self.menu_RandS=Menu(self.menubar)
        self.menu_RandS.add_command(label="rotate",command=self.rotate)
        self.menu_RandS.add_command(label="scale",command=self.scale)
        self.menubar.add_cascade(label="旋转与缩放",menu=self.menu_RandS)
        #创建灰度化子菜单
        self.menu_gray=Menu(self.menubar)
        self.menu_gray.add_command(label="转为灰度图",command=self.rgb2gray)
        self.menu_gray.add_command(label="二值图",command=self.threshold)
        self.menu_gray.add_cascade(label="OTSU",command=self.OTSU)
        self.menubar.add_cascade(label="灰度化",menu=self.menu_gray)
        #创建图像平滑子菜单
        self.menu_noiseremove=Menu(self.menubar)
        self.menu_noiseremove.add_command(label="均值滤波",command=self.mean_remove)
        self.menu_noiseremove.add_command(label="中值滤波",command=self.median_remove)
        self.menu_noiseremove.add_command(label="高斯平滑",command=self.gauss_remove)
        self.menubar.add_cascade(label="图像平滑",menu=self.menu_noiseremove)

        #创建图像增强子菜单
        self.menu_inhance=Menu(self.menubar)
        self.menu_inhance.add_command(label="直方图均衡化",command=self.hist_equalize)
        self.menubar.add_cascade(label="图像增强",menu=self.menu_inhance)
        #在图像增强中创建图像锐化子菜单
        self.submenu_sharper=Menu(self.menu_inhance)
        self.submenu_sharper.add_command(label="Sobel",command=self.inhance_sobel)
        self.submenu_sharper.add_command(label="Laplacian",command=self.inhance_laplacian)
        self.submenu_sharper.add_command(label="Canny",command=self.inhance_Canny)
        self.menu_inhance.add_cascade(label="图像锐化",menu=self.submenu_sharper)
            #在图像增强中创建灰度变换子菜单
        self.submenu_tranlation=Menu(self.menu_inhance)
        self.submenu_tranlation.add_command(label="图像反转",command=self.translation_reverse)
        self.submenu_tranlation.add_command(label="对数变换",command=self.translation_log)
        self.submenu_tranlation.add_command(label="伽马变换",command=self.translation_gamma)
        self.menu_inhance.add_cascade(label="灰度变换",menu=self.submenu_tranlation)
        #创建形态学变换菜单
        self.menu_morphological=Menu(self.menubar)
        self.menu_morphological.add_command(label="腐蚀",command=self.morphology_erode)
        self.menu_morphological.add_command(label="膨胀",command=self.morphology_dailate)
        self.menu_morphological.add_command(label="开运算",command=self.morphology_open)
        self.menu_morphological.add_command(label="闭运算",command=self.morphology_close)
        self.menubar.add_cascade(label="形态学变换",menu=self.menu_morphological)
        # 设置右侧步骤显示区域
        self.right_area=Frame(self.root)
        self.right_area.pack(side=RIGHT,fill=Y)
        self.list_procedure = Listbox(self.right_area)
        self.scollbar=Scrollbar(self.right_area)
        self.scollbar.pack(side=RIGHT,fill=Y)
        self.scollbar.config(command=self.list_procedure.yview)
        self.list_procedure.config(yscrollcommand=self.scollbar.set,height=30)
        self.list_procedure.pack(fill=Y)
        self.botton_see=Button(self.right_area,command=self.review,text="查看").pack()

# 点击响应
    # 打开图片的点击响应
    def openDir(self):
        self.delet_all()
        files = [("PNG图片", "*.png"), ("JPG(JPEG)图片", "*.j[e]{0,1}pg"), ("所有文件", "*")]
        self.file_path_open = filedialog.askopenfilename(title="选择图片", filetypes=files)
        if len(self.file_path_open) is not 0:
            self.im_src = window.cv_imread(self.file_path_open)
            self.show_image(self.im_src)
            self.finish_process(self.im_src.copy(),"read in a image")
    def saveDir(self):
        files = [("PNG图片", ".png"), ("JPG(JPEG)图片", ".j[e]{0,1}pg")]
        self.file_path_save = filedialog.asksaveasfilename(title="选择保存路径",initialfile= "", defaultextension=".png",filetypes=files)
        if len(self.file_path_save) is not 0:
            im = self.pop()
            if im is not None:
                cv.imwrite(self.file_path_save,im)
                simpledialog.messagebox.showinfo("保存成功","路径"+self.file_path_save)
            else:
                simpledialog.messagebox.showerror("尚未选择图片")
    # 图像平滑-均值滤波
    def mean_remove(self):
        im_dist = self.pop()
        if im_dist is None:
            return
        msize = min(im_dist.shape[:2])
        ksize = simpledialog.askinteger("输入平滑操作的卷积核大小", "", initialvalue=3, minvalue=1, maxvalue=msize)
        if ksize is None:
            return
        if ksize%2 is 0:
            simpledialog.messagebox.showerror("参数错误","卷积核大小应为奇数")
            return
        kernel = np.ones((ksize,ksize), np.uint8)
        kernel = kernel/ksize/ksize
        im = cv.filter2D(im_dist,-1,kernel)
        self.finish_process(im,"mean filer")
    # 图像平滑-中值滤波
    def median_remove(self):
        im_dist = self.pop()
        if im_dist is None:
            return
        msize = min(im_dist.shape[:2])
        ksize=None
        ksize = simpledialog.askinteger("输入平滑操作的卷积核大小", "", initialvalue=3, minvalue=1, maxvalue=msize)
        if ksize is None:
            return
        if ksize%2 is 0:
            simpledialog.messagebox.showerror("参数错误","卷积核大小应为奇数")
            return
        im = cv.medianBlur(im_dist,ksize)
        self.finish_process(im, "median filer")
    # 图像平滑-高斯滤波
    def gauss_remove(self):
        im_dist = self.pop()
        if im_dist is None:
            return
        msize = min(im_dist.shape[:2])
        ksize = simpledialog.askinteger("输入平滑操作的卷积核大小", "", initialvalue=3, minvalue=1, maxvalue=msize)
        if ksize is None:
            return
        if ksize % 2 is 0:
            simpledialog.messagebox.showerror("参数错误", "卷积核大小应为奇数")
            return
        sigmaX = simpledialog.askfloat("设定高斯滤波参数sigmaX","",initialvalue=1, minvalue=1, maxvalue=msize)
        if sigmaX is None:
            return
        sigmaY = simpledialog.askfloat("设定高斯滤波参数sigmaY","",initialvalue=sigmaX, minvalue=1, maxvalue=msize)
        if ksize is None:
            return

        im = cv.GaussianBlur(im_dist,(ksize,ksize),sigmaX=sigmaX,sigmaY=sigmaY)
        self.finish_process(im, "gauss filer")
    # 形态学运算-腐蚀
    def morphology_erode(self):
        im_dist = self.pop()
        if im_dist is None:
            return
        if len(im_dist.shape) is 3:
            self.rgb2gray()
            im_dist = cv.cvtColor(im_dist, cv.COLOR_RGB2GRAY)
        if np.logical_and(im_dist>0,im_dist<255).any():
            #需要二值化
            self.OTSU()
            _,im_dist = cv.threshold(im_dist,0,255,cv.THRESH_OTSU)
        msize=min(im_dist.shape[:2])
        ksize = simpledialog.askinteger("输入腐蚀操作的卷积核大小","",initialvalue=3,minvalue=1,maxvalue=msize)
        if ksize is None:
            return
        time = simpledialog.askinteger("输入操作次数","",initialvalue=1,minvalue=1)
        if time is None:
            return
        ksize = (ksize,ksize)
        kernel=np.ones(ksize,np.uint8)
        im = cv.morphologyEx(im_dist,cv.MORPH_ERODE,kernel=kernel,iterations=time)
        self.finish_process(im,"erode x"+str(time)+" ksize:"+str(ksize))
    # 形态学运算-膨胀
    def morphology_dailate(self):
        im_dist = self.pop()
        if im_dist is None:
            return
        if len(im_dist.shape) is 3:
            self.rgb2gray()
            im_dist = cv.cvtColor(im_dist, cv.COLOR_RGB2GRAY)
        if np.logical_and(im_dist > 0, im_dist < 255).any():
            # 需要二值化
            self.OTSU()
            _, im_dist = cv.threshold(im_dist, 0, 255, cv.THRESH_OTSU)
        msize = min(im_dist.shape[:2])
        ksize = simpledialog.askinteger("输入膨胀操作的卷积核大小", "", initialvalue=3,minvalue=1,maxvalue=msize)
        if ksize is None:
            return
        time = simpledialog.askinteger("输入操作次数", "", initialvalue=1, minvalue=1)
        if time is None:
            return
        ksize = (ksize, ksize)
        kernel = np.ones(ksize, np.uint8)
        im = cv.morphologyEx(im_dist, cv.MORPH_DILATE, kernel=kernel, iterations=time)
        self.finish_process(im, "dailate x" + str(time) + " ksize:" + str(ksize))
    #形态学运算-开
    def morphology_open(self):
        im_dist = self.pop()
        if im_dist is None:
            return
        if len(im_dist.shape) is 3:
            self.rgb2gray()
            im_dist = cv.cvtColor(im_dist, cv.COLOR_RGB2GRAY)
        if np.logical_and(im_dist > 0, im_dist < 255).any():
            # 需要二值化
            self.OTSU()
            _, im_dist = cv.threshold(im_dist, 0, 255, cv.THRESH_OTSU)
        msize = min(im_dist.shape[:2])
        ksize = simpledialog.askinteger("输入开运算的卷积核大小", "", initialvalue=3,minvalue=1,maxvalue=msize)
        if ksize is None:
            return
        time = simpledialog.askinteger("输入操作次数", "", initialvalue=1, minvalue=1)
        if time is None:
            return
        ksize = (ksize, ksize)
        kernel = np.ones(ksize, np.uint8)
        im = cv.morphologyEx(im_dist, cv.MORPH_OPEN, kernel=kernel, iterations=time)
        self.finish_process(im, "open x" + str(time) + " ksize:" + str(ksize))
    # 形态学运算-闭运算
    def morphology_close(self):
        im_dist = self.pop()
        if im_dist is None:
            return
        if len(im_dist.shape) is 3:
            self.rgb2gray()
            im_dist = cv.cvtColor(im_dist, cv.COLOR_RGB2GRAY)
        if np.logical_and(im_dist > 0, im_dist < 255).any():
            # 需要二值化
            self.OTSU()
            _, im_dist = cv.threshold(im_dist, 0, 255, cv.THRESH_OTSU)
        msize = min(im_dist.shape[:2])
        ksize = simpledialog.askinteger("输入闭运算的卷积核大小", "", initialvalue=3, minvalue=1, maxvalue=msize)
        if ksize is None:
            return
        time = simpledialog.askinteger("输入操作次数", "", initialvalue=1, minvalue=1)
        if time is None:
            return
        ksize = (ksize, ksize)
        kernel = np.ones(ksize, np.uint8)
        im = cv.morphologyEx(im_dist, cv.MORPH_CLOSE, kernel=kernel, iterations=time)
        self.finish_process(im, "close x" + str(time) + " ksize:" + str(ksize))
    # sobel算子
    def inhance_sobel(self):
        im_dist = self.pop()
        if im_dist is None:
            return
        if len(im_dist.shape) is 3:
            self.rgb2gray()
            im_dist=cv.cvtColor(im_dist,cv.COLOR_RGB2GRAY)

        x = cv.Sobel(im_dist, cv.CV_16S, 1, 0)
        y = cv.Sobel(im_dist, cv.CV_16S, 0, 1)

        absX = cv.convertScaleAbs(x)
        absY = cv.convertScaleAbs(y)

        dst = cv.addWeighted(absX, 0.5, absY, 0.5, 0)
        dst = np.uint8(dst)
        self.finish_process(dst,"sobel edge")

    # laplacian算子
    def inhance_laplacian(self):
        im_dist = self.pop()
        if im_dist is None:
            return
        if len(im_dist.shape) is 3:
            self.rgb2gray()
            im_dist=cv.cvtColor(im_dist,cv.COLOR_RGB2GRAY)
        dst = cv.Laplacian(im_dist,cv.CV_16S)
        dst = np.uint8(dst)
        self.finish_process(dst,"laplacian edge")


    # Canny算子
    def inhance_Canny(self):
        im_dist = self.pop()
        if im_dist is None:
            return
        if len(im_dist.shape) is 3:
            self.rgb2gray()
            im_dist=cv.cvtColor(im_dist,cv.COLOR_BGR2GRAY)
        dst = cv.Canny(im_dist,125,255)
        dst = np.uint8(dst)
        self.finish_process(dst, "canny edge")

    # 图像反转点击相应
    def translation_reverse(self):
        im_dist=self.pop()
        if im_dist is None:
            return
        if len(im_dist) is 3:
            b,g,r=cv.split(im_dist)
            r = 255-r
            g = 255-g
            b = 255-b
            im_dist=cv.merge([b,g,r])
        else:
            im_dist = 255-im_dist
        self.finish_process(im_dist,"reverse image")
    # 伽玛变换点击响应
    def translation_gamma(self):
        im_dist = self.pop()
        if im_dist is None:
            return
        c = simpledialog.askfloat("设置伽马变换的参数c","s=cr^γ")
        if c is None:
            return
        gamma = simpledialog.askfloat("设置伽马变换的参数γ","s=cr^γ")
        if gamma is None:
            return
        if len(im_dist) is 3:
            for k in range(3):
                im = im_dist[:,:,k]
                max_pixel=np.max(im)
                im=np.uint8(
                    c*np.power(im/max_pixel,gamma)*max_pixel
                )
                im_dist[:,:,k]=im
        else:
            max_pixel = np.max(im_dist)
            im_dist = np.uint8(
                    c * np.power(im_dist / max_pixel, gamma) * max_pixel
            )
        self.finish_process(im_dist,"gamma with coefficient c="+str(c)+" gamma="+str(gamma))
    # 对数变换点击相应
    def translation_log(self):
        im_dist = self.pop()
        if im_dist is None:
            return
        c = simpledialog.askfloat("设置对数变换的参数c","s = c*log(1+r)")
        if c is None:
            return
        if len(im_dist) is 3:
            for k in range(3):
                im = im_dist[:,:,k]
                max_pixel=np.max(im)
                im=np.uint8(
                    (c*np.log(1+im)-c*np.log(1+0))/\
                        (c*np.log(1+max_pixel)-c*np.log(1+0))*max_pixel
                            )
                im_dist[:,:,k]=im
        else:
            max_pixel = np.max(im_dist)
            im_dist = np.uint8(
                    (c * np.log(1 + im_dist) - c * np.log(1 + 0)) / \
                      (c * np.log(1 + max_pixel) - c * np.log(1 + 0)) * max_pixel
            )
        self.finish_process(im_dist,"log with coefficient "+str(c))

    # 旋转点击相应
    def rotate(self):
        im_dist=self.pop()
        if im_dist is None:
            return
        (h, w) = im_dist.shape[:2]
        degree = simpledialog.askfloat("输入旋转角度",
                              "顺时针为负，逆时针为正",
                              initialvalue=90,
                              maxvalue=180,
                              minvalue=-180)
        if degree is None:
            return
        hNew = int(w * fabs(sin(radians(degree))) + h * fabs(cos(radians(degree))))
        wNew = int(h * fabs(sin(radians(degree))) + w * fabs(cos(radians(degree))))
        center = (w//2,h//2)

        M = cv.getRotationMatrix2D(center, degree, 1.0)
        M[0, 2] += (wNew - w) / 2
        M[1, 2] += (hNew - h) / 2
        im = cv.warpAffine(im_dist, M, ( wNew,hNew), borderValue=(255, 255, 255))
        self.finish_process(im,"rotate "+str(degree)+"degree")
    # 缩放点击响应
    def scale(self):
        im_dist=self.pop()
        if im_dist is None:
            return
        (h, w) = im_dist.shape[:2]
        rate = simpledialog.askfloat("输入缩放比例",
                                     "此缩放为等比例缩放，缩放操作之后，超过屏幕显示范围的话将会将显示大小调整至合适的范围，原图大小已按照缩放比例改变",
                                     initialvalue=1,
                                     minvalue=0.1)
        if rate is None:
            return
        im_dist = cv.resize(im_dist,dsize=(0,0),fx=rate,fy=rate)
        self.finish_process(im_dist,"scale to initial's "+str(rate))
    # 直方图均衡化点击响应
    def hist_equalize(self):
        im_dist=self.pop()
        if im_dist is None:
            return
        if im_dist.shape[-1] is 3:
            r,g,b=cv.split(im_dist)
            r = cv.equalizeHist(r)
            g = cv.equalizeHist(g)
            b = cv.equalizeHist(b)
            im_dist = cv.merge((r,g,b))
        else:
            im_dist = cv.equalizeHist(im_dist)
        self.finish_process(im_dist,"hist_equalize")
    # 灰度化点击相应
    def rgb2gray(self):
        im_dist=self.pop()
        if im_dist is None:
            return
        if len(im_dist.shape) is 3:
            im_dist = cv.cvtColor(im_dist,cv.COLOR_RGB2GRAY)
            self.finish_process(im_dist,"RGB to GRAY")
        else:
            self.finish_process(im_dist,"GRAY to GRAY")
    # 二值化点击相应
    def threshold(self):
        im_dist=self.pop()
        if im_dist is None:
            return
        thresh_min = simpledialog.askinteger("输入最小阈值",
                                "该值以下的像素值将被置为0",
                                initialvalue=125,
                                maxvalue=255,
                                minvalue=1)
        if thresh_min is None:
            return
        thresh_max = simpledialog.askinteger("输入最大阈值",
                                             "该值以上的像素值将被置为0",
                                             initialvalue = 255,
                                             maxvalue=255,
                                             minvalue=1)
        if thresh_max is None:
            return
        if thresh_max <= thresh_min:
            simpledialog.messagebox.showerror("最大阈值需要大于最大阈值")
            return
        if len(im_dist.shape) is 3:
            im_dist = cv.cvtColor(im_dist,cv.COLOR_RGB2GRAY)
            self.rgb2gray()
        _,im_dist=cv.threshold(im_dist,thresh_min,thresh_max,cv.THRESH_BINARY)
        self.finish_process(im_dist,"thresh "+str(thresh_min)+" to "+str(thresh_max))
    # 最佳阈值分割点击响应
    def OTSU(self):
        im_dist = self.pop()
        if im_dist is None:
            return
        if len(im_dist.shape) is 3:
            im_dist = cv.cvtColor(im_dist,cv.COLOR_RGB2GRAY)
            self.rgb2gray()
        thresh,im_dist=cv.threshold(im_dist,0,255,cv.THRESH_OTSU)
        self.finish_process(im_dist,"thresh OTSU "+str(thresh))
# 工具方法
    # 图像处理后的维护操作
    def finish_process(self,im,name:str):
        self.show_image(im)#显示图像
        self.im_dist.append(im)#将图像缓存
        self.list_procedure.insert(END,name)#将步骤缓存
        self.root.update()#更新

    # 获取用户要查看的缓存图像位置，并显示对应的图片
    def review(self):
        indexs=self.list_procedure.curselection()
        self.show_image(self.im_dist[indexs[0]])

    # 图片堆栈弹出
    def pop(self,delete=False):
        if len(self.im_dist) == 0:
            return None
        if delete == False:
            # 在图像处理时使用，仅用于获取最新一次处理后的图像的副本
            return self.im_dist[-1].copy()
        else:
            # 在调用返回上一步操作时使用，用于返回图像并删除缓存
            self.list_procedure.delete(END)
            return self.im_dist.pop()
    # 显示图片
    def show_image(self,im,newsize:tuple=None):
        self.im_show = im.copy()
        if newsize is None:
            (imsizecol,imsizerow) = self.get_reshape_size(self.im_show)
        else:
            (imsizecol,imsizerow) = newsize
        self.im_show = cv.resize(self.im_show, (imsizecol, imsizerow))
        self.root.geometry("%sx%s"%(int(self.width_max*4/5),int(self.height_max*4/5)))
        self.im_tk = window.im_np2im_tk(self.im_show)
        self.imageforshow.config(height = imsizerow,width=imsizecol)
        self.imageforshow.create_image(0,0,anchor=NW,image=self.im_tk)
        self.root.update()

    # 返回上一次处理的图片，并删除缓存
    def back(self):
        self.pop(delete=True)
        self.show_image(self.pop())

    # 获取能够最好地显示的图片大小
    def get_reshape_size(self,im):
        if len(im.shape) is 3:
            imsizerow, imsizecol, _ = im.shape
        else:
            imsizerow, imsizecol = im.shape

        #将图片调整至最佳大小
        if imsizecol > self.width_max*0.7 or imsizerow > self.height_max*0.7:
            if imsizecol > imsizerow:
                imsizerow = int(self.width_max*0.7*imsizerow/imsizecol)
                imsizecol = int(self.width_max*0.7)

            if imsizecol <= imsizerow:
                imsizecol = int(self.height_max*0.7*imsizecol/imsizerow)
                imsizerow = int(self.height_max*0.7)

        return (imsizecol,imsizerow)

    def im_np2im_tk(im):
        # 改变三通道排列顺序并将图像转换为可显示的类型
        if len(im.shape) == 3:
            im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        img = Image.fromarray(im)
        imTk = ImageTk.PhotoImage(image=img)
        return imTk

    def delet_all(self):
        if len(self.im_dist) is not 0:
            ok = simpledialog.messagebox.askokcancel("注意","打开新图片将删除所有已有的图片，确定要继续打开吗")
            if ok is None:
                return
            if ok is False:
                return
            self.im_dist.clear()
            self.list_procedure.delete(0,END)

    def cv_imread(file_path=""):
        #编码格式转换
        cv_img = cv.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
        return cv_img

w = window()
