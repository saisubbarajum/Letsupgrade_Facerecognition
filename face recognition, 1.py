import cv2
import numpy as np
image1 = cv2.imread("images1.jpg")
image2 = cv2.imread("images2.jpg")
image3 = cv2.imread("images3.jpg")
image4 = cv2.imread("images4.jpg")
image5 = cv2.imread("images5.jpg")
image6 = cv2.imread("images6.jpg")

# press 1 ,2 ,3 ,4 ,5 for changes the background images

# image 1

cap = cv2.VideoCapture(0)

while True:
    sucess, frame = cap.read()
    if not sucess:
        print("video is not opened")
        break
    
    
    blended_image = cv2.addWeighted(frame, 0.8, image1, 0.2, gamma=0.2)
    cv2.imshow("blended image",blended_image)
   
        
        
    k = cv2.waitKey(50)
    

    if k == ord("1"):

        # image 2

        cap = cv2.VideoCapture(0)

        while True:
            sucess, frame = cap.read()
            if not sucess:
                print("video is not opened")
                break
            blended_image = cv2.addWeighted(frame, 0.7, image2, 0.3, gamma=0.2)
            cv2.imshow("blended image",blended_image)
            k = cv2.waitKey(50)
            if k == ord("2"):

                # image 3

                cap = cv2.VideoCapture(0)

                while True:
                        sucess, frame = cap.read()
                        if not sucess:
                            print("video is not opened")
                            break
                        
                        
                        blended_image = cv2.addWeighted(frame, 0.7, image3, 0.3, gamma=0.2)
                        cv2.imshow("blended image",blended_image)
            
                        k = cv2.waitKey(10)

                        if k == ord("3"):

                            # image 4
                            cap = cv2.VideoCapture(0)
                            while True:
                                    sucess, frame = cap.read()
                                    if not sucess:
                                        print("video is not opened")
                                        break
                                    
                                    
                                    blended_image = cv2.addWeighted(frame, 0.7, image4, 0.3, gamma=0.2)
                                    cv2.imshow("blended image",blended_image)
                                    k = cv2.waitKey(10)
                                    if k == ord("4"):
                                        # image 5
                                        cap = cv2.VideoCapture(0)
                                        while True:
                                                sucess, frame = cap.read()
                                                if not sucess:
                                                    print("video is not opened")
                                                    break
                                                
                                                
                                                blended_image = cv2.addWeighted(frame, 0.7, image5, 0.3, gamma=0.2)
                                                cv2.imshow("blended image",blended_image)
                                                k = cv2.waitKey(10)
                                                if k == ord("5"):
                                                    #image 6
                                                    cap = cv2.VideoCapture(0)
                                                    while True:
                                                            sucess, frame = cap.read()
                                                            if not sucess:
                                                                print("video is not opened")
                                                                break
                                                                
                                                                
                                                            blended_image = cv2.addWeighted(frame, 0.7, image6, 0.3, gamma=0.2)
                                                            cv2.imshow("blended image",blended_image)
                                                            k = cv2.waitKey(10)
                                                            if k == ord("q"):
                                                                break

                                                    break
                                        break
                                    elif k == ord("q"):
                                        break
                            break              
                        
                        elif k == ord("q"):
                             break
            
                break     

            elif k == ord("q"):
                break
                
        break


    elif k == ord("q"):
                break

    

cap.release()
cv2.destroyAllWindows()