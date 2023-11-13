from cProfile import label
from tkinter import *
from tkinter import filedialog as fd
from tkinter import font
from tkinter.ttk import Progressbar
from tkinter import messagebox as mg
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
import time
SMOOTHING_RADIUS=50
root=Tk()
root.geometry("600x250")
root.resizable(0,0)
root.title("Video Stabilizer")
file_filter=(("MP4 Files","*.mp4"),('AVI Files','*.avi'))
def in_click():
    try:
        global infile
        infile=[]
        file=fd.askopenfile(title="Select Input video",initialdir='/',filetypes=file_filter)
        video_input.delete(0,END)
        video_input.insert(0,file.name)
        infile=file.name.split('/') 
    except AttributeError as e:
        pass 
def out_click():
    try:
        global outfile
        outfile=[]
        file=fd.askdirectory(title="Select Output folder",initialdir='/')
        video_ouput.delete(0,END)
        video_ouput.insert(0,file)
        outfile=file.split('/')
        save=infile[len(infile)-1].split('.')
        outfile.append(save[0]+"_stabalized_output.mp4")
    except IndexError as e:
        pass
def sumbit():
    try:
        if (len(infile)!=0 and len(outfile)!=0):
            
            t=0;speed=1;pre=StringVar()
            done.config(text="                        ",fg="grey")
            precent=Label(root,textvariable=pre)
            loading=Progressbar(root,orient=HORIZONTAL,length=400,mode="determinate")
            loading.place(relx=0.15,rely=0.6)
            precent.place(relx=0.45,rely=0.7)
        
            while(t<100):
                time.sleep(0.05)
                loading['value']+=(speed/100)*100
                t+=speed
                pre.set(str(int((t/100)*100))+"%")
                root.update_idletasks()
            stabalise()
            done.config(text="Video is Saved",fg="green")
        else:
            mg.showerror("Error","Input/Output is Path Not Selected")
    except NameError as e:
        mg.showerror("Error","Input/Output is Path Not Selected")
def exit1():
    root.destroy()
title1=Label(root,text="Video Stabilizer",font=("Arial black",20))
video_input=Entry(root,width=46,font=('Arial',15))
video_ouput=Entry(root,width=46,font=('Arial',15))
done=Label(root,fg="green",font=("Ariel"))
    

b1=Button(root,text="Browse",font=('Arial',10),command=in_click)
b2=Button(root,text="Browse",font=('Arial',10),command=out_click)
su=Button(root,text="Submit",font=('Arial',10),command=sumbit)
ex=Button(root,text="EXIT",bg="Red",fg="white",font=('Arial',10),width=6,command=exit1)

b1.place(rely=0.2)
b2.place(rely=0.35)
su.place(relx=0.9,rely=0.51)
ex.place(relx=0.9,rely=0.63)

title1.place(relx = 0.5, rely = 0, anchor = N)
video_input.place(relx=0.1,rely=0.2)
video_ouput.place(relx=0.1,rely=0.35)
done.place(relx=0.6,rely=0.8)
def movingAverage(curve, radius): 
	window_size = 2 * radius + 1
	
	f = np.ones(window_size)/window_size 
	 
	curve_pad = np.lib.pad(curve, (radius, radius), 'edge') 
	
	curve_smoothed = np.convolve(curve_pad, f, mode='same') 
	
	curve_smoothed = curve_smoothed[radius:-radius]
	
	return curve_smoothed 

def smooth(trajectory): 
	smoothed_trajectory = np.copy(trajectory) 
	
	for i in range(3):
		smoothed_trajectory[:,i] = movingAverage(trajectory[:,i], radius=SMOOTHING_RADIUS)
 
	return smoothed_trajectory

def fixBorder(frame):
     s = frame.shape
     
     T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.04)
     frame = cv2.warpAffine(frame, T, (s[1], s[0]))
     return frame
def stabalise():
    
    cp = cv2.VideoCapture("/".join(infile))

    
    n_frames = int(cp.get(cv2.CAP_PROP_FRAME_COUNT))

    
    

    width = int(cp.get(cv2.CAP_PROP_FRAME_WIDTH)) 
    height = int(cp.get(cv2.CAP_PROP_FRAME_HEIGHT))

    

    
    fps = cp.get(cv2.CAP_PROP_FPS)

    

    
    out = cv2.VideoWriter('/'.join(outfile), 0x7634706d, fps, (width, height))

    
    _, prev = cp.read()

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    transforms = np.zeros((n_frames-1, 3), np.float32) 

    for i in range(n_frames-2):
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=20, qualityLevel=0.01, minDistance=5, blockSize=3)

        succ, curr = cp.read()

        if not succ:
            break

        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)


        
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)


        idx = np.where(status==1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]
        assert prev_pts.shape == curr_pts.shape 


        m = cv2.estimateRigidTransform(prev_pts, curr_pts, fullAffine=False) 

        dx = m[0,2]
        dy = m[1,2]

        
        da = np.arctan2(m[1,0], m[0,0])
            
        transforms[i] = [dx,dy,da] 

        prev_gray = curr_gray
        

        
    trajectory = np.cumsum(transforms, axis=0) 

    smoothed_trajectory = smooth(trajectory)
    difference = smoothed_trajectory - trajectory
    transforms_smooth = transforms + difference

    
    cp.set(cv2.CAP_PROP_POS_FRAMES, 0) 
    
    for i in range(n_frames-2):
        
        success, frame = cp.read() 
        if not success:
            break

        
        dx = transforms_smooth[i,0]
        dy = transforms_smooth[i,1]
        da = transforms_smooth[i,2]

       
        m = np.zeros((2,3), np.float32)
        m[0,0] = np.cos(da)
        m[0,1] = -np.sin(da)
        m[1,0] = np.sin(da)
        m[1,1] = np.cos(da)
        m[0,2] = dx
        m[1,2] = dy

        
        frame_stabilized = cv2.warpAffine(frame, m, (width,height))

       
        frame_stabilized = fixBorder(frame_stabilized) 

        
        frame_out = cv2.hconcat([frame, frame_stabilized])

        
        if(frame_out.shape[1] > 854): 
            frame_out = cv2.resize(frame_out, (frame_out.shape[1]//3, frame_out.shape[0]//3))
        
        cv2.imshow("Before and After", frame_out)
        out.write(frame_stabilized)
        cv2.waitKey(10)
        
    cp.release()
    cv2.destroyAllWindows()
    
    out.release()
root.mainloop()