import cv2
from cv2 import imshow
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from os import listdir
import queue

image_row = 0 
image_col = 0

# visualizing the mask (size : "image width" * "image height")
def mask_visualization(M):
    mask = np.copy(np.reshape(M, (image_row, image_col)))
    plt.figure()
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')

# visualizing the unit normal vector in RGB color space
# N is the normal map which contains the "unit normal vector" of all pixels (size : "image width" * "image height" * 3)
def normal_visualization(N):
    # converting the array shape to (w*h) * 3 , every row is a normal vetor of one pixel
    N_map = np.copy(np.reshape(N, (image_row, image_col, 3)))
    # Rescale to [0,1] float number
    N_map = (N_map + 1.0) / 2.0
    plt.figure()
    plt.imshow(N_map)
    plt.title('Normal map')

# visualizing the depth on 2D image
# D is the depth map which contains "only the z value" of all pixels (size : "image width" * "image height")
def depth_visualization(D):
    D_map = np.copy(np.reshape(D, (image_row,image_col)))
    # D = np.uint8(D)
    plt.figure()
    plt.imshow(D_map)
    plt.colorbar(label='Distance to Camera')
    plt.title('Depth map')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')

# convert depth map to point cloud and save it to ply file
# Z is the depth map which contains "only the z value" of all pixels (size : "image width" * "image height")
def save_ply(Z,filepath):
    Z_map = np.reshape(Z, (image_row,image_col)).copy()
    data = np.zeros((image_row*image_col,3),dtype=np.float32)
    # let all point float on a base plane 
    baseline_val = np.min(Z_map)
    Z_map[np.where(Z_map == 0)] = baseline_val
    for i in range(image_row):
        for j in range(image_col):
            idx = i * image_col + j
            data[idx][0] = j
            data[idx][1] = i
            data[idx][2] = Z_map[image_row - 1 - i][j]
    # output to ply file
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    o3d.io.write_point_cloud(filepath, pcd,write_ascii=True)

# show the result of saved ply file
def show_ply(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    o3d.visualization.draw_geometries([pcd])

# read the .bmp file
def read_bmp(filepath):
    global image_row
    global image_col
    image = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
    image_row , image_col = image.shape
    return image

def get_I_and_L(test_case):
    # read the light source file
    light_source = open("test/" + test_case + "/LightSource.txt", "r")
    light_list = light_source.read().split('\n') 

    imgs = []
    # read all the bmp files and data 
    for i in range(len(light_list) - 1):
        # light_list[i][0]: file name of picture, light_list[i][1]: light source
        light_list[i] = light_list[i].split(": ")    
        # read the bmp files
        file_path = "test/" + test_case +"/" + str(light_list[i][0]) + ".bmp"
        img_read = np.reshape(read_bmp(file_path), (1, image_row*image_col))                              
        imgs.append(np.reshape(img_read, (1, image_row*image_col)))
        # change the light source data into the format I want
        light_list[i] = light_list[i][1][1:-1].split(",")
        for j in range(len(light_list[i])):
            light_list[i][j] = int(light_list[i][j])
    
    # initialize I and L matrix
    img_num = len(imgs)
    I = np.zeros((img_num, image_row*image_col))
    L = np.zeros((img_num, 3))
    # get the I and L matrix
    for i in range(img_num):
        I[i] = imgs[i]
        L[i] = np.array(light_list[i])
        L[i] = L[i] / np.linalg.norm(L[i])
    
    return I, L 

# calculate normal
def normal_estimate(I, L):
    L_transpose = np.transpose(L)
    KdN = np.linalg.inv(L_transpose.dot(L)).dot(L_transpose).dot(I)
    KdN = np.transpose(KdN) # to make it in the correct direction
    # calculate the norm of each normal vector
    KdN_norm = np.reshape(np.repeat(np.linalg.norm(KdN, axis = 1), 3), KdN.shape)
    KdN_norm[KdN_norm == 0] = 1
    # normalize each normal vector
    N = KdN / KdN_norm
    return N 

def BFS_integrate(Gradient, mask, start_x, start_y): 
    pixel_queue = queue.Queue()
    pixel_queue.put(((start_x, start_y), 0))   # element =  (position, depth)
    pixel_visit = np.zeros((image_row, image_col))
    pixel_visit[start_y, start_x] = 1
    surface = np.zeros((image_row, image_col))
    
    while not pixel_queue.empty():
        current_pixel = pixel_queue.get()
        x = current_pixel[0][0]
        y = current_pixel[0][1]
        parent_depth = current_pixel[1]      

        # top
        if y-1 >= 0 and pixel_visit[y-1, x] == 0 and mask[y-1, x]:
            depth = parent_depth - Gradient[y,x,1]
            surface[y-1, x] = depth
            pixel_queue.put(((x, y-1), depth))
            pixel_visit[y-1, x] = 1        
        # down
        if y+1 < image_row and pixel_visit[y+1, x] == 0 and mask[y+1, x]:
            depth = parent_depth + Gradient[y,x,1]
            surface[y+1, x] = depth
            pixel_queue.put(((x, y+1), depth))
            pixel_visit[y+1, x] = 1        
        # left
        if x-1 >= 0 and pixel_visit[y, x-1] == 0 and mask[y, x-1]:
            depth = parent_depth - Gradient[y,x,0]
            surface[y, x-1] = depth
            pixel_queue.put(((x-1, y), depth))
            pixel_visit[y, x-1] = 1        
        #right
        if x+1 < image_col and pixel_visit[y, x+1] == 0 and mask[y, x+1]:
            depth = parent_depth + Gradient[y,x,0]
            surface[y, x+1] = depth
            pixel_queue.put(((x+1, y), depth))
            pixel_visit[y, x+1] = 1 
                 
    surface += abs(np.min(surface))
    return surface
    

# reconstruct the surface
def surface_reconstruct(N): 
    # construct a mask
    mask = np.linalg.norm(N, axis = 1)
    mask[mask != 0] = 1
    mask = np.reshape(mask, (image_row, image_col))
    mask_visualization(mask)   
    # calculate the gradient
    N_x = N[:, 0]
    N_y = N[:,1]
    N_z = N[:, 2]
    N_z_x = N_z[N_x==0] = 1
    N_z_y = N_z[N_y==0] = 1
    N_x = -N_x / N_z_x
    N_y = N_y / N_z_y
    Gradient = np.transpose(np.array([N_x, N_y])) 
    Gradient = np.reshape(Gradient,(image_row, image_col, 2))
    
    surface1 = BFS_integrate(Gradient, mask, int(image_col / 2), int(image_row / 4))
    surface2 = BFS_integrate(Gradient, mask, int(image_col / 2), int(image_row / 4 * 3))
    surface3 = BFS_integrate(Gradient, mask, int(image_col / 4), int(image_row / 2))
    surface4 = BFS_integrate(Gradient, mask, int(image_col / 4 * 3), int(image_row / 2))
    surface5 = BFS_integrate(Gradient, mask, int(image_col / 2), int(image_row / 2))
    
    weight1 = np.reshape(np.repeat(np.linspace(1, 0, image_row), image_col), (image_row, image_col))
    weight2 = np.reshape(np.repeat(np.linspace(0, 1, image_row), image_col), (image_row, image_col))
    weight3 = np.transpose(np.reshape(np.repeat(np.linspace(1, 0, image_col), image_row), (image_col, image_row)))
    weight4 = np.transpose(np.reshape(np.repeat(np.linspace(0, 1, image_col), image_row), (image_col, image_row)))
    weight5 = np.ones((image_row, image_col)) / 2
        
    # get the average of all the surfaces 
    surface = (surface1*weight1 + surface2*weight2 + surface3*weight3 + surface4*weight4 + surface5*weight5) * 0.6
    # apply the mask on the surface    
    surface = surface * mask
    return surface
    


if __name__ == '__main__':
    # get all the test case under the directory
    path = "./test"
    dir = listdir(path)
    
    for test_case in dir:         
        # get the I and L matrix
        I, L = get_I_and_L(test_case)        

        # calculate the N matrix and visualize it
        N = normal_estimate(I, L)
        normal_visualization(N)
        
        # calculate the depth and visualize it
        Z = surface_reconstruct(N)
        depth_visualization(Z)  
        
        # showing the windows of all visualization function
        plt.show()      
        
        # save the ply file and show it
        save_ply(Z, "./"+ test_case +".ply")
        show_ply("./"+ test_case +".ply")