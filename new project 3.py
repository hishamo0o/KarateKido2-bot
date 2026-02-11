import cv2 as cv
import mss
import numpy as np
from collections import Counter
import pygetwindow as gw
import pyautogui
import time

game_intro_template = cv.imread('images\\template.png')
blue_branch_image = cv.imread('branches\\blue_branch.png')

n2 = cv.imread('numbers\\2.png', 0)
n3 = cv.imread('numbers\\3.png', 0)
n4 = cv.imread('numbers\\4.png', 0)

n2_halfw_halfh = cv.resize(n2 , (n2.shape[0]//2  , n2.shape[1]//2))
n3_halfw_halfh = cv.resize(n3 , (n3.shape[0]//2  , n3.shape[1]//2))
n4_halfw_halfh = cv.resize(n4 , (n4.shape[0]//2  , n4.shape[1]//2))

# _ , n2_kernel =cv.threshold(n2 , 128 , 1 , cv.THRESH_BINARY)
# _ , n3_kernel =cv.threshold(n3, 128 , 1 , cv.THRESH_BINARY)
# _ , n4_kernel =cv.threshold(n4 , 128 , 1 , cv.THRESH_BINARY)

# _ , n2_kernel_halfw_halfh =cv.threshold(n2_halfw_halfh , 128 , 1 , cv.THRESH_BINARY)
# _ , n3_kernel_halfw_halfh =cv.threshold(n3_halfw_halfh, 128 , 1 , cv.THRESH_BINARY)
# _ , n4_kernel_halfw_halfh =cv.threshold(n4_halfw_halfh , 128 , 1 , cv.THRESH_BINARY)

kernels = {'2' :[n2 , n2_halfw_halfh] ,
           '3' : [n3, n3_halfw_halfh] ,
           '4': [n4 , n4_halfw_halfh]
           }

sift = cv.SIFT_create()

template_keypoints , template_descriptors = sift.detectAndCompute(game_intro_template , None)

min_x , max_x = -1 , -1

tree_min_x , tree_max_x , tree_min_y , tree_max_y , single_obs_height , tree_mid_point = -1 , -1 , -1 , -1 , -1 , -1

game_detected = False

window = None

tree_mask = None

tree_color = None

g_game_stream = None

padding = 0

#0 for left , 1 for right

def get_vertical_lines(img ,padding=10 , convert_to_gray = True):
    # Add padding:
    img = img[:, padding: img.shape[1] - padding]

    if convert_to_gray:
        img = cv.cvtColor(img , cv.COLOR_BGR2GRAY)

    # Apply Canny edge detection with adjusted parameters
    edges = cv.Canny(img, 50, 200)
    
    # Apply Probabilistic Hough Line Transform with adjusted parameters
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=150, minLineLength=80, maxLineGap=20)
    
    # Check if lines were detected
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) < 5:  # Filter near-vertical lines
                cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # cv.imshow('Detected Lines', img)
    # cv.imshow('Canny Edges', edges)
    
    min_x = float('inf') 
    max_x = float('-inf') 
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) < 5:
                if min(x1, x2) < min_x:
                    min_x = min(x1, x2)
                if max(x1, x2) > max_x:
                    max_x = max(x1, x2)
                
    if min_x == float('inf') or max_x == float('-inf'):
        print('min_x and max_x are not set')

        return 0 , img.shape[0]
    return int(min_x)+padding, int(max_x)+padding

def check_for_horizontal_lines(image , draw = False):
    image = cv.resize(image , (300 , 100))
    # Apply Canny edge detection
    edges = cv.Canny(image, 10, 100)

    # Apply Probabilistic Hough Line Transform with adjusted parameters
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=25 , maxLineGap=20)

    # Initialize a flag to indicate whether horizontal lines are found
    found_horizontal_lines = False

    image = cv.cvtColor(image , cv.COLOR_GRAY2RGB)

    # Filter and find horizontal lines
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 5:  # Filter out non-horizontal lines
                # Draw the horizontal line on the image
                if draw:
                    cv.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                found_horizontal_lines = True

    return found_horizontal_lines, image


def is_game_window(img):
    img_keypoints , img_descriptors = sift.detectAndCompute(img , None)
    # Use FLANN based matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=7)
    search_params = dict(checks=100)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    # Match descriptors 
    matches = flann.knnMatch(template_descriptors, img_descriptors, k=2)
    # Apply ratio test to filter good matches 
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    # Check if enough matches are found
    if len(good_matches) > 100:
    #      # Draw matches 
    #     img_matches = cv.drawMatches(game_intro_template, template_keypoints, img, img_keypoints, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #     cv.imshow("Matches", img_matches)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        return True
    else:
        return False

def calc_most_frequent_color(img):
    
    if game_stream is None :
        print('error could not get game stream ')
        img = get_game_stream()

    # Reshape the image to be a list of pixels
    pixels = img.reshape(-1, 3)

    if len(pixels) ==0:
        print('pixels are empty')
        img = get_game_stream()
        pixels = img.reshape(-1,3)

    # Convert to a list of tuples
    pixels_list = [tuple(pixel) for pixel in pixels]

    # Count the frequency of each color
    color_counts = Counter(pixels_list)

    # Get the most common color
    most_common_color = color_counts.most_common(1)[0][0]
    # tree_color = most_common_color
    # print(f'most common color:{most_common_color}')
    
    return most_common_color

def calc_mask_using_color(img,color , apply_close = True  , apply_open = True, tolerance=15):
    
    # Create a mask for the target color
    lower_bound = np.array([max(0, c - tolerance) for c in color], dtype=np.uint8)
    upper_bound = np.array([min(255, c + tolerance) for c in color], dtype=np.uint8)

    mask = cv.inRange(img, lower_bound, upper_bound)

    # Apply the mask to the image
    masked_image = cv.bitwise_and(img, img, mask=mask)

    gray = cv.cvtColor(masked_image, cv.COLOR_BGR2GRAY)
    
    if apply_close or apply_open:
        blurred = cv.GaussianBlur(gray , ksize=(15,15 ), sigmaX=1, sigmaY=1)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (21, 21))
        res = blurred
        # Apply the morphological close operation
        if apply_close:
            res = cv.morphologyEx(res, cv.MORPH_CLOSE, kernel )
            res = cv.morphologyEx(res, cv.MORPH_CLOSE, kernel )
        if apply_open:
            res = cv.morphologyEx(res, cv.MORPH_CLOSE, kernel )

        # tree_mask = closed_image
        # cv.imshow('tree mask' , tree_mask)
        return res
    return gray
    
def get_tree_min_max(game_stream ) :
    global tree_min_x , tree_max_x , tree_min_y , tree_max_y , single_obs_height , tree_mid_point  , tree_color , tree_mask , padding


    tree_min_x , tree_max_x = get_vertical_lines(game_stream , padding=50,convert_to_gray=True)

    # cv.imshow('tree min max' , game_stream[:,tree_min_x :tree_max_x,:])

    tree_color =calc_most_frequent_color(game_stream[:,tree_min_x :tree_max_x,:])
    tree_mask =calc_mask_using_color(game_stream , tree_color)

    non_black_pixels = np.argwhere(tree_mask > 0)

    tree_min_y = np.min(non_black_pixels[:,0])
    tree_max_y = np.max(non_black_pixels[:,0])
    tree_max_y -=0 #padding

    single_obs_height = int((tree_max_y - tree_min_y) / 7)
    print(f'single obsticle height:{single_obs_height}')
    tree_mid_point = int((tree_max_x + tree_min_x)/2)

    print(f'max x : {tree_max_x} , min x : {tree_min_x} , min y :{tree_min_y} , max y : {tree_max_y}')
    # cv.imshow('closed cropped' , tree_mask)
    # [tree_min_y :tree_max_y , min_x :max_x]

def get_tree_obs_region(index_of_obs , width =15):

    if index_of_obs ==0:
        start = 0 
        end = int(.75 * single_obs_height)
    else:
        start = int((index_of_obs - .1) * single_obs_height)
        end = int((index_of_obs + .65) * single_obs_height)

    return width , g_game_stream[start: end, tree_min_x - width:tree_max_x +width,:]

def handle_number(original_obs_image):
    
    original_obs_gray = cv.cvtColor(original_obs_image , cv.COLOR_BGR2GRAY)

    binary_image = cv.Canny(original_obs_gray ,110 , 255 )

    image_h, image_w = binary_image.shape

    res = np.zeros_like(binary_image)

    for key in kernels:
        for kernel in kernels[key]:

            kernel_h , kernel_w = kernel.shape
            if not  kernel_h < image_h and kernel_w < image_w:
                continue
            for y in range(image_h - kernel_h+1):
                for x in range (image_w - kernel_w +1):

                    roi = binary_image[y : y + kernel_h , x : x + kernel_w]
                    
                    res = cv.bitwise_and(kernel , roi)
                    intersection = np.sum(res)
                    kernel_sum = np.sum(kernel)

                    # Check if the difference is below a certain threshold
                    if intersection / kernel_sum >= 0.55:  # Adjust the threshold as needed

                        res = res
                        return int(key)                     
    return 0            

def handle_obsticle(index_of_obs):
    numbers_to_handle = {}

    glass_found = False
    glass_res = {}
    res = []
    for i, j in enumerate(index_of_obs):
        width, original_obs = get_tree_obs_region(j)

        if not glass_found:
            glass_found = check_for_white_color(original_obs)
            if glass_found:
                to_show = cv.resize(original_obs , (300 , 300))
                cv.imshow('glass' , to_show)
                print(f'found glass at {j}')
                glass_res[j] = 2

        original_obs = original_obs[:, width:original_obs.shape[1] - width, :]

        if j == 0:
            continue

        if i - 1 >= 0 and index_of_obs[i - 1] == j - 1 and i + 1 < len(index_of_obs) and index_of_obs[i + 1] == j + 1:
            numbers_to_handle[j] = original_obs
            continue

        if i + 1 == len(index_of_obs) and index_of_obs[i - 1] == j - 1 and i - 2 >= 0 and not index_of_obs[i - 2] == j - 2:
            numbers_to_handle[j] = original_obs
            continue

        if len(index_of_obs) == 2 and i - 1 >= 0 and index_of_obs[i - 1] == j - 1:
            numbers_to_handle[j] = original_obs

    if len(glass_res)>0:
        # print(f'glass found at {glass_res.keys()}')
        for i in glass_res:
            res.append((i , 2))

    if len(numbers_to_handle) > 0:
        for i in numbers_to_handle.keys():
            number = handle_number(numbers_to_handle[i])
            if number != 0:
                # print(f'number {number} found at {i}')
                if i not in glass_res:
                    res.append((i, number))
                else:
                    res.append((i, number + 1))
            else:
                if i in glass_res:
                    res.append((i, 2))
                print(f'Could not detect the number at {i}')  # Handle when number is not found

    if len(glass_res) > 0:
        for i in glass_res.keys():
            found = False
            if len(res) != 0:
                for j in res:
                    if i == j[0]:
                        found = True
                        break
            if not found:
                res.append((i,2))

    return res


def get_tree_colored_image(index):
    if index ==0:
        start = 0 
        end = int(.75 * single_obs_height)
    else:
        start = int((index - .2) * single_obs_height)
        end = int((index + .9) * single_obs_height)

    return g_game_stream[start:end , tree_min_x:tree_max_x]

def get_right_side_colored(index , width =-1):
    if width ==-1:
        width = 2*(tree_max_x - tree_min_x)
    if index ==0:
        start = 0 
        end = int(.75 * single_obs_height)
    else:
        start = int((index - .2) * single_obs_height)
        end = int((index + .9) * single_obs_height)
    return g_game_stream[start : end ,tree_max_x : tree_max_x+width ]

def get_left_side_colored(index , width =-1):
    if width ==-1:
        width = 2*(tree_max_x - tree_min_x)
    if index ==0:
        start = 0 
        end = int(.75 * single_obs_height)
    else:
        start = int((index - .2) * single_obs_height)
        end = int((index + .9) * single_obs_height)
    return g_game_stream[start : end ,tree_min_x - width : tree_min_x ]

def handle_blue_branches(right_side_colored , left_side_colored):

    blue_color = calc_most_frequent_color(blue_branch_image)

    right_mask = calc_mask_using_color(right_side_colored , blue_color , tolerance=12)
    left_mask = calc_mask_using_color(left_side_colored , blue_color , tolerance=12)

    right_branch_found , right_side_mask = check_for_horizontal_lines(right_mask , draw=True)
    left_branch_found , left_side_mask = check_for_horizontal_lines(left_mask , draw=True)

    if right_branch_found or left_branch_found:
        return  right_branch_found , left_branch_found , right_side_mask , left_side_mask
    return False , False , None , None

def handle_colored_branches(tree_obs_colored , right_side_colored , left_side_colored , ratio):
    color = calc_most_frequent_color(tree_obs_colored)

    right_side_mask = calc_mask_using_color(right_side_colored , color , apply_close=True, apply_open=False , tolerance=10)
    left_side_mask = calc_mask_using_color(left_side_colored , color , apply_close=True, apply_open=False,tolerance=10)

    # right_side_mask = cv.resize(right_side_mask ,(300 , 100) , interpolation=cv.INTER_CUBIC)
    # left_side_mask = cv.resize(left_side_mask ,(300 , 100) , interpolation=cv.INTER_CUBIC)

    right_branch_found , right_side_mask = check_for_horizontal_lines(right_side_mask , draw=True)
    left_branch_found , left_side_mask = check_for_horizontal_lines(left_side_mask , draw=True)

    if not right_branch_found or not left_branch_found and ratio < .35:
        right_branch_found_v , left_branch_found_v , right_side_mask_v , left_side_mask_v = handle_blue_branches(right_side_colored , left_side_colored)

        if right_branch_found_v or left_branch_found_v :

            right_branch_found = right_branch_found_v
            left_branch_found = left_branch_found_v
            right_side_mask = right_side_mask_v
            left_side_mask = left_side_mask_v

        return right_branch_found , left_branch_found , right_side_mask , left_side_mask
    return right_branch_found , left_branch_found , right_side_mask , left_side_mask 

def check_for_white_color(image):
    # Convert to HSV color space
    hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV)

    # Define the color range for pure white
    lower_white = np.array([0, 0, 240])
    upper_white = np.array([180, 30, 255])

    # Create a mask to detect pure white color
    mask = cv.inRange(hsv, lower_white, upper_white)

    # Find contours of the white regions
    # contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Draw rectangles around the detected regions
    # for contour in contours:
        # x, y, w, h = cv.boundingRect(contour)
        # cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # print('1')

    # Check if pure white color is detected
    if cv.countNonZero(mask) > 1:
        return True
    else:
        return False

def check_colored_branch(index_of_branch , ratio):

    colored_tree_obs = get_tree_colored_image(index_of_branch)
    colored_righ_side = get_right_side_colored(index_of_branch)
    colored_left_side = get_left_side_colored(index_of_branch)

    right_found , left_found , right_side_mask , left_side_mask =handle_colored_branches(colored_tree_obs , colored_righ_side , colored_left_side , ratio)

    return right_found , left_found , right_side_mask , left_side_mask

def get_frames_move():
    res =[]
    numbers_to_handle =[]
    width =  2*(tree_max_x - tree_min_x)
    low_counter = 0
    for i in range(7):
        if i ==0:
            start = 0 
            end = int(.75 * single_obs_height)
        else:
            start = int((i - .2) * single_obs_height)
            end = int((i + .9) * single_obs_height)

        tree_obsticles_i =tree_mask[ start  :  end  ,
                                     tree_min_x:tree_max_x]
        right_side_i =tree_mask[ start  : end,
                                 tree_max_x: tree_max_x +width ]
        left_side_i =tree_mask[start : end,
                                tree_min_x -width:tree_min_x  ]
        
        tree_obs_count = np.count_nonzero(tree_obsticles_i)
        tree_obs_total = tree_obsticles_i.shape[0] *  tree_obsticles_i.shape[1]
        ratio = round(tree_obs_count / tree_obs_total ,3)
        
        right_found , right_side_i=check_for_horizontal_lines(right_side_i , draw=True)
        left_found , left_side_i=check_for_horizontal_lines(left_side_i, draw=True)

        if ratio < .5  :
            numbers_to_handle.append(i)
            if not (right_found or left_found) and ratio <.1:
                low_counter+=1
                right_found_v , left_found_v , right_side_v , left_side_v = check_colored_branch(i , ratio)

                if right_found_v or left_found_v:

                    right_found = right_found_v
                    left_found = left_found_v

                    right_side_i = right_side_v
                    left_side_i = left_side_v
        
        res.append([i , right_found or left_found,  ratio , left_found , right_found])

        tree_obsticles_i = cv.cvtColor(tree_obsticles_i , cv.COLOR_GRAY2RGB)

        tree_obsticles_i = cv.resize(tree_obsticles_i ,(300 , 100) , interpolation=cv.INTER_CUBIC)

        cv.putText(tree_obsticles_i , f'{ratio}' , (50, 50) ,cv.FONT_HERSHEY_COMPLEX , fontScale=1 , color=(0,255,0),thickness=3)

        cv.imshow(f'tree obs{i}' , tree_obsticles_i)
        cv.imshow(f'right {i}' , right_side_i)
        cv.imshow(f'left {i}' , left_side_i)

    if low_counter >=5:
        return None, None

    xor_res =[]
    false_counter = 0
    for i , j in enumerate(res):
        if i == 0 :
            inversed = not j[1]
            xor_res.append(j[1] ^ inversed)
        if i-1 >= 0:
            prev =  res[i-1]
            temp = prev[1]^ j[1]
            if not temp:
                false_counter +=1
            xor_res.append(temp)
    # print(xor_res)
    
    lantern =-1

    for i in range(len(xor_res)):

        if i-1 >0 :

            if not xor_res[i] and xor_res[i-1]:
                lantern = i-2

    numbers = []

    if len(numbers_to_handle)>0:
        numbers = handle_obsticle(numbers_to_handle)

    moves = {}

    for i in range(7):
        move = ''
        number_of_hits = -1
        current = res[i]

        if current[1] :

            if current[3]:

                move = 'right'

            if current[4]:

                move = 'left'

        if not current[1] and not i==0:
            prev = moves[i-1]
            move = prev[1]
        if len(numbers)>0:
            for index , values in enumerate(numbers):

                currnt_number = values 
                if i == currnt_number[0]:
                    number_of_hits = currnt_number [1]
                    break
        if number_of_hits ==-1 :
            number_of_hits =1

        moves[i] = (number_of_hits , move)

    if lantern !=-1 :
        branch_have_lantern = res[lantern]
        if not branch_have_lantern:
            print('detected alntern but there is no branch')

    # print(moves)
    return moves , lantern
   
    

def get_frames_moves():
    obs_to_handle =[]

    number_and_maybe_glass_without_branches_found =[]

    glass_or_number_with_branches_found = []

    found_branch_last_loop = False

    width = 2*(tree_max_x - tree_min_x)

    res = {}

    for i in range(7):
        rb_found = False
        lb_found = False
        move =''
        p_found_branch_last_loop = found_branch_last_loop

        if i ==0:
            start = 0 
            end = int(.75 * single_obs_height)
        else:
            start = int((i - .2) * single_obs_height)
            end = int((i + .9) * single_obs_height)

        tree_obsticles_i =tree_mask[ start  :  end  ,
                                     tree_min_x:tree_max_x]
        right_side_i =tree_mask[ start  : end,
                                 tree_max_x: tree_max_x +width ]
        left_side_i =tree_mask[start : end,
                                tree_min_x -width:tree_min_x  ]

        tree_obs_count = np.count_nonzero(tree_obsticles_i)
        right_side_count = np.count_nonzero(right_side_i)
        left_side_count = np.count_nonzero(left_side_i)

        tree_obs_total = tree_obsticles_i.shape[0] *  tree_obsticles_i.shape[1]
        rights_side_total = right_side_i.shape[0] * right_side_i.shape[1]
        left_side_total = left_side_i.shape[0] * left_side_i.shape[1]

        ratio = round(tree_obs_count / tree_obs_total ,3)
        ratio_r =round(right_side_count / rights_side_total,3)
        ratio_l = round(left_side_count / left_side_total,3)

        tree_obsticles_i = cv.resize(tree_obsticles_i ,(300 , 100) , interpolation=cv.INTER_CUBIC)
        right_side_i = cv.resize(right_side_i ,(300 , 100) , interpolation=cv.INTER_CUBIC)
        left_side_i = cv.resize(left_side_i ,(300 , 100) , interpolation=cv.INTER_CUBIC)
        if not found_branch_last_loop :
            # _ , tree_obsticles_i=check_for_horizontal_lines(tree_obsticles_i)
            right_found , right_side_i=check_for_horizontal_lines(right_side_i , draw=True)
            left_found , left_side_i=check_for_horizontal_lines(left_side_i, draw=True)
            if right_found :
                # res[f'{i}'] = ('left' , 1)
                move = 'left'
                rb_found = True
                found_branch_last_loop = True
            if left_found:
                # res[f'{i}'] = ('right',1)
                move = 'right'
                lb_found = True
                found_branch_last_loop = True
            elif not( right_found or left_found):
                if not i ==0:
                    number_and_maybe_glass_without_branches_found.append(i)
                    # res[f'{i}'] = ''
                    # print(f'there must be a branch at{i}')
                found_branch_last_loop = False
        else:
            t = list(res[f'{i-1}'])
            move  = t[0]
            found_branch_last_loop=False
            
        tree_obsticles_i = cv.cvtColor(tree_obsticles_i , cv.COLOR_GRAY2RGB)
        # right_side_i = cv.cvtColor(right_side_i , cv.COLOR_GRAY2RGB)
        # left_side_i = cv.cvtColor(left_side_i , cv.COLOR_GRAY2RGB)

        cv.putText(tree_obsticles_i , f'{ratio}' , (50, 50) ,cv.FONT_HERSHEY_COMPLEX , fontScale=1 , color=(0,255,0),thickness=3)
        cv.putText(right_side_i , f'{ratio_r}' , (50, 50) ,cv.FONT_HERSHEY_COMPLEX , fontScale=1 , color=(0,255,0),thickness=3)
        cv.putText(left_side_i , f'{ratio_l}' , (50, 50) ,cv.FONT_HERSHEY_COMPLEX , fontScale=1 , color=(0,255,0),thickness=3)
        
        # print(f'tree obs:' , tree_obsticles_i.shape[0] * tree_obsticles_i.shape[1])
        # print(f'right_side_i:' , right_side_i.shape[0] * right_side_i.shape[1])
        # print(f'left_side_i:' , left_side_i.shape[0] * left_side_i.shape[1])
        if ratio < .6  and ((not(p_found_branch_last_loop) and (lb_found or rb_found)) or (p_found_branch_last_loop)):
            glass_or_number_with_branches_found.append(i)
            # print(f'appended at {i}')
            # obs_to_handle. append(i)

        res[f'{i}'] = (move , 1)


        cv.imshow(f'tree obs{i}' , tree_obsticles_i)
        cv.imshow(f'right {i}' , right_side_i)
        cv.imshow(f'left {i}' , left_side_i)

    # if not res_branches_found:
    #         res_branches_found = handle_obsticles(glass_or_number_with_branches_found ,handle_colored_branches=False)
    #         common_keys = set(res_branches_found.keys() & set(res.keys()))
    #         if not common_keys:
    #             for key in common_keys:
    #                 if not res[f'{key}']:
    #                     res[f'{key}']= 

    # obs_handled_res = {}

    # if  len(obs_to_handle) >0:
    #     # print('glass or number found')
    #     obs_handled_res=handle_obsticle(obs_to_handle)
    #     if len(obs_handled_res)>0:
    #         for obs_index in obs_handled_res.keys():
    #             if obs_index in res.keys() and not res[f'{obs_index}'] == 'error':
    #                 t=list(res[f'{obs_index}'])
    #                 t[1] +=obs_handled_res[f'{obs_index}']
    #                 res[f'{obs_index}']= tuple(t)
    #     else:
    #         print('called handle obs and returned noth')
    print(res)
    print(f'glass or numbers with branches :{glass_or_number_with_branches_found}')
    


    


def press_key(key , count=1 ):
    global window
    if window == None :
        print('window is none ')
        return
    
    x =int( window.left + window.width / 2)
    y =int( window.top + window.height / 2)
    offset = 100 if key == 'right' else -100
    x += offset
    while count>0:
        pyautogui.leftClick(x, y)
        count-=1


def get_game_stream():
    screenshot = sct.grab(bbox)
    img = cv.cvtColor(np.array(screenshot), cv.COLOR_RGBA2RGB)
    game_stream = img[212:int(img.shape[0] *.9), min_x:max_x, :]
    return game_stream

def get_tree_stream(game_stream):
    global tree_min_y , tree_min_y , tree_min_x , tree_max_x

    if tree_min_x ==-1 or tree_max_x == -1 or tree_max_y ==-1 or tree_min_y ==-1 :
        print('error , can\'t get tree min and max')
        print(f'max x : {tree_max_x} , min x : {tree_min_x} , min y :{tree_min_y} , max y : {tree_max_y}')

    return game_stream[tree_min_y : tree_max_y , tree_min_x :tree_max_x,:]


# Main capturing loop
with mss.mss() as sct:
    window = gw.getWindowsWithTitle('Play games, WIN REAL REWARDS! | GAMEE')[0]
    bbox = {'top': window.top, 'left': window.left, 'width': window.width, 'height': window.height}
    # to crop the game only
    window.activate()
    time.sleep(.01)
    stream =np.array(sct.grab(bbox))
    min_x, max_x = get_vertical_lines(stream ,padding= 50, convert_to_gray=False)
    prev_time = time.time()
    moves_counter = 0
    moves = {}
    while True:
        if tree_color!=None:
            g_game_stream = get_game_stream()
            tree_mask = calc_mask_using_color(g_game_stream , tree_color)
        if moves_counter >0:
            current_move = moves[moves_counter]
            if not lantern ==-1 :
                print(f'lantern :{lantern } , coutner :{moves_counter}')
                if lantern == moves_counter-2:
                    print(f'detected lantern at {lantern} , counter {moves_counter}')
                    single_move = 'left' if current_move[1] == 'right' else 'right'
                    press_key(single_move ,1)
                    press_key(current_move ,current_move[0] -1)
            else:
                press_key(current_move[1] ,current_move[0] )
            moves_counter-=1


        game_stream = get_game_stream()
        if not game_detected :

            game_detected = is_game_window(game_stream)
            if game_detected :

                print('gmae detected')
                #perform a click or 3 .
                press_key('right' , count =3 )
                #get tree min and max
                time.sleep(.1)
                g_game_stream = get_game_stream()
                # print(f'window width : {max_x - min_x}')
                # print(f'widow height : {game_stream.shape[0] -212}')
                # print(f'ratio :{(max_x-min_x) / (game_stream.shape[0]-212)}')
                get_tree_min_max(g_game_stream)
                # print(f'tree width : {tree_max_x - tree_min_x}')
                # print(f'tree height : {tree_max_y -tree_min_y}')
                # print(f'tree ratio :{(tree_max_x-tree_min_x) / (tree_max_y -tree_min_y)}')
                if tree_min_x ==-1 or tree_max_x == -1 or tree_max_y ==-1 or tree_min_y ==-1 :
                    
                    print('error , can\'t get tree min and max')
                    print(f'max x : {tree_max_x} , min x : {tree_min_x} , min y :{tree_min_y} , max y : {tree_max_y}')

        if game_detected and moves_counter ==0:
            time.sleep(.4)
            # cv.imshow('detected tree:' , g_game_stream[:,tree_min_x :tree_max_x ,:])
            g_game_stream = get_game_stream()
            tree_mask = calc_mask_using_color(g_game_stream , tree_color , tolerance=18)
            #move , lantern
            moves , lantern =get_frames_move()#move : number , side
            # print(f'len of moves = {len(moves)}')
            if moves ==None and lantern ==-2:
                print('level up!!')
                # time.sleep(1)
                continue
            moves_counter = 6
            print(moves)
    

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

# Release everything
cv.destroyAllWindows()

