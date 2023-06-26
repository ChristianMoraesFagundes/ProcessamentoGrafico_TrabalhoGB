import cv2
import numpy as np
img_counter = 0
cam = cv2.VideoCapture(0)
img = cv2.VideoCapture(0)
#overlay stickers
def overlay(background, foreground, x_offset=None, y_offset=None):
    bg_h, bg_w, bg_channels = background.shape
    fg_h, fg_w, fg_channels = foreground.shape

    assert bg_channels == 3, f'background image should have exactly 3 channels (RGB). found:{bg_channels}'
    assert fg_channels == 4, f'foreground image should have exactly 4 channels (RGBA). found:{fg_channels}'

    # center by default
    if x_offset is None: x_offset = (bg_w - fg_w) // 2
    if y_offset is None: y_offset = (bg_h - fg_h) // 2

    w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
    h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

    if w < 1 or h < 1: return

    # clip foreground and background images to the overlapping regions
    bg_x = max(0, x_offset)
    bg_y = max(0, y_offset)
    fg_x = max(0, x_offset * -1)
    fg_y = max(0, y_offset * -1)
    foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]
    background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]

    # separate alpha and color channels from the foreground image
    foreground_colors = foreground[:, :, :3]
    alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0

    # construct an alpha_mask that matches the image shape
    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

    # combine the background with the overlay image weighted by alpha
    composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask

    # overwrite the section of the background image that has been updated
    background[bg_y:bg_y + h, bg_x:bg_x + w] = composite
    out = background.copy()
    return out 

#define the events for the
# mouse_click.
def mouse_click(event, x, y, 
                flags, param):
      
    # to check if left mouse 
    # button was clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        global img_counter 

        if param[0] == '1':
            sticker = cv2.imread(f'stickers/{param[1]}.png', cv2.IMREAD_UNCHANGED)
            #captura e salva a imagem original
            img_name = "opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            
            imgCapture = cv2.imread(f'opencv_frame_{img_counter}.png')
            print(imgCapture.shape) 
           
            out = overlay(imgCapture,sticker, x, y) 

            #salva imagem com sticker
            img_name_sticker = "opencv_sticker_{}.png".format(img_counter)
            cv2.imwrite(img_name_sticker, out)
            print("{} written!".format(img_name_sticker))
            
            img_counter+=1
            cv2.imshow('image', out)
        elif param[0] == '2':
            #captura e salva a imagem original
            img_name = "opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))

            imgCapture = cv2.imread(f'opencv_frame_{img_counter}.png')
            img_filter = ""
            if param[1] == '1':
                #Gray Filter
                img_filter =  cv2.cvtColor(imgCapture, cv2.COLOR_BGR2GRAY)
            elif param[1] == '2':
                #Blue detect
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                # define range of blue color in HSV
                lower_blue = np.array([110,50,50])
                upper_blue = np.array([130,255,255])
                img_filter  = cv2.inRange(hsv, lower_blue, upper_blue)
            elif param[1] == '3':
                #Blue Channel
                # set green and red channels to 0
                img_filter = imgCapture.copy()
                img_filter[:, :, 1] = 0
                img_filter[:, :, 2] = 0
            elif param[1] == '4':
                #Red Channel
                # set green and red channels to 0
                img_filter = imgCapture.copy()
                img_filter[:, :, 0] = 0
                img_filter[:, :, 1] = 0
            elif param[1] == '5':
                #Green Channel
                # set green and red channels to 0
                img_filter = imgCapture.copy()
                img_filter[:, :, 0] = 0
                img_filter[:, :, 2] = 0
            elif param[1] == '6':
                #Blur
                imgGrayscale =  cv2.cvtColor(imgCapture, cv2.COLOR_BGR2GRAY)
                img_filter = cv2.GaussianBlur(imgGrayscale,(15,15),0)
            elif param[1] == '7':     
                #Bordas Sobel
                imgGrayscale =  cv2.cvtColor(imgCapture, cv2.COLOR_BGR2GRAY)
                imgBlurred = cv2.GaussianBlur(imgGrayscale,(1,1),0)
                img_filter = cv2.Sobel(imgBlurred,50,100)
            elif param[1] == '8':     
                #Bordas Canny
                imgGrayscale =  cv2.cvtColor(imgCapture, cv2.COLOR_BGR2GRAY)
                imgBlurred = cv2.GaussianBlur(imgGrayscale,(1,1),0)
                img_filter = cv2.Canny(imgBlurred,50,100)
            elif param[1] == '9':
                #Erosão
                kernel = np.ones((5,5), np.uint8) 
                img_filter = cv2.erode(imgCapture, kernel, iterations=1) 
            elif param[1] == '10':
                #Dilatação
                kernel = np.ones((5,5), np.uint8) 
                img_filter = cv2.dilate(imgCapture, kernel, iterations=1) 
            elif param[1] == '11':
                #Aquarela
                winter = cv2.cvtColor(imgCapture,cv2.COLORMAP_WINTER)
                kernel = np.ones((5,5), np.uint8) 
                img_filter = cv2.erode(winter,kernel, iterations=2)
            #salva imagem com filtro
            img_name = "opencv_filter_{}.png".format(img_counter)
            cv2.imwrite(img_name, img_filter)
            print("{} written!".format(img_name))
            cv2.imshow("Imagem Com Filtro",img_filter)
    
        

        
        
        
  
          
cv2.namedWindow("Trabalho")

#seleção de sticker ou filtro
print('Deseja adicionar um stickers ou oum filtro? 1 - sticker / 2 - filtro')
imgAdd=''    
tipo = input()
if tipo == '1':
    print('Selecione um stickers')
    print('1 - festa')  
    print('2 - óculos escuros')       
    print('3 - hang loose')  
    print('4 - Paz')  
    print('5 - Buccaneer')  
    imgAdd = input()
elif tipo =='2':
    print('Selecione um stickers')
    print('1 - Grayscale') # grayscale 
    print('2 - Blue Detect') # blue detect   
    print('3 - Blue Channel') # blue channel
    print('4 - Red Channel') # red channel 
    print('5 - Green Chanel') # green channel 
    print('6 - Gaussian Blur') # Gaussian blur
    print('7 - Bordas Sobel') # bordas sobel   erroooooooooooooo   
    print('8- Bordas Canny')  # bordas canny
    print('9 - Erosão')  # erosão
    print('10 - Dilatação') 
    print('11 - Blue Vitral')  
    imgAdd = input()
param = [tipo,imgAdd]
cv2.setMouseCallback('Trabalho', mouse_click,param)


while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("Trabalho", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()