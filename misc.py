import pandas as pd
import numpy as np
import random
import cv2

'''
gen_weights(): It generates the values (weights) that are going to modify the current position of the white ball, these weights depend on the method.
If you want to create a new method, you just need to add an elif statement and the new logic of the movement to create the new 'new_x', 'new_y' variables and return them as a list.

parameters:
- r: Radius of where the pixel can 'jump' in the next position. The bigger the radius, the crazier the movement.
- method: Behaviour of the white pixel, currently there are 4 types of behaviours (Explained above gen_video())
- res: It recieves a single value of the frame's dimension (frames_res[0], this is one of the reasons it may not work well with a non squared dimension)
- corr: Parameter that dictates how SEMI_RANDOM_CORR works, I'm not sure what happens if corr=0 or corr=1 => use it between 0 < corr < 1.
- weights_anteriores: Previous weights used to modify the previous white ball coordinates, this parameter is only used in method='SEMI_RANDOM_CORR'
- sentido: Parameter only used in method='LINE', ignore it if you don't use this method.
'''

def gen_weights(r, method, res, corr, weights_anteriores, sentido):
    new_x = 0 # new x value to add to the x coordinate
    new_y = 0 # new y value to add to the y coordinate

    if method == 'SEMI_RANDOM': # Chooses a random value from radius.
        weight_range = [x for x in range(-r, r + 1)] # if r=2 => weight_range = [-2, -1, 0, 1, 2]
        new_x = random.choice(weight_range) # Chooses a random value from weight_range
        new_y = random.choice(weight_range) # Chooses a random value from weight_range

    elif method == 'RANDOM': # Chooses a random value from within the frame
        weight_range = [x for x in range(-res, res + 1)] # if res=50 => weight_range = [-50, -49, ..., 0, ..., 49, 50]
        new_x = random.choice(weight_range) # Chooses a random value from weight_range
        new_y = random.choice(weight_range) # Chooses a random value from weight_range

    elif method == 'SEMI_RANDOM_CORR': # Chooses a random value from within the radius depending on the previous weights and a 'corr' probability.
        weight_range = [x for x in range(-r, r + 1)] # if r=2 => weight_range = [-2, -1, 0, 1, 2]
        x_pos_prob_range = [corr / r for x in range(1, r + 1)] # if r=2 and corr=0.6 => x_pos_prob_range = [0.3, 0.3]
        x_neg_prog_range = [(1 - corr) / (r+1) for x in range(-r, 1)] # if r=2 and corr=0.6 => x_pos_prob_range = [0.4/3, 0.4/3, 0.4/3]


        x_pos_range = x_neg_prog_range + x_pos_prob_range # [0.4/3, 0.4/3, 0.4/3, 0.3, 0.3] ; This list gives greater probability for the weight_range positive values
        x_neg_range = x_pos_prob_range + x_neg_prog_range # [0.3, 0.3, 0.4/3, 0.4/3, 0.4/3] ; This list gives greater probability for the weight_range negative values

        y_pos_prob_range = [corr / r for x in range(1, r + 1)]  # if r=2 and corr=0.6 => y_pos_prob_range = [0.3, 0.3]
        y_neg_prog_range = [(1 - corr) / (r + 1) for x in range(-r, 1)] # if r=2 and corr=0.6 => y_pos_prob_range = [0.4/3, 0.4/3, 0.4/3]

        y_pos_range = y_neg_prog_range + y_pos_prob_range # [0.4/3, 0.4/3, 0.4/3, 0.3, 0.3] ; This list gives greater probability for the weight_range positive values
        y_neg_range = y_pos_prob_range + y_neg_prog_range # [0.3, 0.3, 0.4/3, 0.4/3, 0.4/3] ; This list gives greater probability for the weight_range negative values

        # The lists could be reused for x and y coordinates.

        # These if's check for the previous vector direction. This allows the positive/negative correlation.
        if weights_anteriores[0] > 0:
            new_x = random.choices(weight_range, weights=x_pos_range)[0]
        elif weights_anteriores[0] <= 0:
            new_x = random.choices(weight_range, weights=x_neg_range)[0]

        if weights_anteriores[1] > 0:
            new_y = random.choices(weight_range, weights=y_pos_range)[0]
        elif weights_anteriores[1] <= 0:
            new_y = random.choices(weight_range, weights=y_neg_range)[0]

    # Method for creating a line, not that interesting.
    elif method == 'LINE':
        new_x = (r*sentido)
        new_y = 0

    return [new_x, new_y]

'''
video(): Generates the coordinates of the white ball, it also makes sure those coordinates stay within the frame's boundaries.

parameters:
- x: Starting position of the white ball in 'x' axis.
- y: Staring position of the white ball in 'y' axis.
- frames: Number of frames in the video (frames_no).
- r: Radius of where the pixel can 'jump' in the next position. The bigger the radius, the crazier the movement.
- res: It recieves a single value of the frame's dimension (frames_res[0], this is one of the reasons it may not work well with a non squared dimension)
- size_bolita: Size of the white pixel, it needs to have a centre, this means that it won't work well with size (2, 2), usual acceptable values are (1, 1), (3, 3) or (5, 5).
- method: Behaviour of the white pixel, currently there are 4 types of behaviours (Explained above gen_video())
- corr: Parameter that dictates how SEMI_RANDOM_CORR works, I'm not sure what happens if corr=0 or corr=1 => use it between 0 < corr < 1.
'''
def video(x, y, frames, r, res, size_bolita, method, corr):
    diff = np.ceil((size_bolita[0]/2)) # Length from centre of white ball to its border.
    positions = [[x, y]] # List to store all white ball positions
    current_pos = [x, y] # List that stores current position.
    weights_anteriores = [0, 0] # Previous weights, basically how much we added to the previous positions, so that they became the next positions.
    c_frames = 0 # Number of frames generated.
    sentido = 1 # I think this variable was for the method=LINE, you can ignore it.

    while c_frames != frames: # It stops when we generate the number of frames we wanted.
        next = gen_weights(r, method, res, corr, weights_anteriores, sentido) # Calls function gen_weights(), it generates how much we need to sum (+/-) to the current coordinates.
        current_pos[0] += next[0] # Sums new x weight into current x position.
        current_pos[1] += next[1] # Sums new y weight into current y position.

        # This 'if' takes care that the white ball doesnt escape the frame's boundaries, if it does, then it backtracks and subtracts the previously added weights.
        if current_pos[0] >= ((res-1)-(diff-1)) or current_pos[0] <= (0+(diff-1)) or current_pos[1] >= ((res-1)-(diff-1)) or current_pos[1] <= (0+(diff-1)):
            current_pos[0] -= next[0]
            current_pos[1] -= next[1]
            sentido *= -1
        # If the white ball doesn't escape the frames boundaries, then it continues generating frames.
        else:
            weights_anteriores = next # It saves the current weights as "previous" weights (useful for the next frame)
            positions.append([current_pos[1], current_pos[0]]) # It saves the new position of the white ball.
            c_frames += 1

    return positions # Once it generates all the coordinates for the video, it returns them.

'''
create_video(): Using the given coordinates of the white ball, it creates the white ball in the frame.

parameters:
- positions: coordinates of white ball.
- res: It recieves a single value of the frame's dimension (frames_res[0], this is one of the reasons it may not work well with a non squared dimension)
- size_bolita: Size of the white pixel, it needs to have a centre, this means that it won't work well with size (2, 2), usual acceptable values are (1, 1), (3, 3) or (5, 5).
'''

def create_video(positions, res, size_bolita):
    frames = [] # list for storing the frames that are ready.

    for x in positions: # Iterates through the coordinates.
        start_pos_x = x[0] - (np.ceil(size_bolita[0]/2)-1) # x coordinate for top left of white ball.
        start_pos_y = x[1] + (np.ceil(size_bolita[0]/2)-1) # y coordinate for top left of white ball
        frame = np.zeros((res, res), dtype=int) # creates a frame filled with zeros.

        for i in range(size_bolita[0]): # Iterates through length of white ball
            for j in range(size_bolita[1]): # Iterates through width of white ball
                frame[int(start_pos_x+i)][int(start_pos_y-j)] = 255 # Creates pixel for the white ball.

        frames.append(frame) # Saves frame

    return frames # Returns frames

'''
create_grayscale_video(): Uses the generated frames to create a .mp4 video.

parameters:
- frames_list: frames from create_video()
- output_file: path to save the .mp4.
- fps: frames per second. more fps -> smoother, but also the video lasts less, to compensate you can create more frames.
'''

def create_grayscale_video(frames_list, output_file, fps=28):
    height, width = frames_list[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height), isColor=False)
    c = 0

    for frame in frames_list:
        frame_uint8 = np.uint8(frame)  # Convert the frame to uint8 (8-bit grayscale)
        out.write(frame_uint8)
        c+=1

    out.release()
    return frames_list

'''
gen_video(): Recieves the initial parameters and calls the necessary functions to create the generated coordinates (video()),
fill the frames with white pixels (create_video()) and then export the frames into an .mp4 video (create_grayscale_video())

parameters:
- r: Radius of where the pixel can 'jump' in the next position. The bigger the radius, the crazier the movement.
- size_bolita: Size of the white pixel, it needs to have a centre, this means that it won't work well with size (2, 2), usual acceptable values are (1, 1), (3, 3) or (5, 5).
- method: Behaviour of the white pixel, currently there are 4 types of behaviours:
    - LINE: The white pixel moves in a line.
    - RANDOM: The white pixel can choose any position within the frame at random.
    - SEMI_RANDOM: The white pixel can choose any position within its radius at random.
    - SEMI_RANDOM_CORR: If parameter corr >= 0.5, the white pixel is more likely to choose a next position with the same vector direction as the previous position. 
                        If parameter corr < 0.5, the white pixel is less likely to choose a next position with the same vector direction as the previous position.
- corr: Parameter that dictates how SEMI_RANDOM_CORR works, I'm not sure what happens if corr=0 or corr=1 => use it between 0 < corr < 1.
- frame_res: resolution of the video, I recommend keeping it squared (Example 50x50), non-squared may work badly since I didn't design it with that in mind.
- starting_pos: Staring position of the white ball, it needs to be within the video resolution. (obviously)
- frames_no= Number of frames to be generated.
- save_path= Directory to save the video. 
'''
def gen_video(r, size_bolita, method='SEMI_RANDOM', corr=0.5, frames_res=(50, 50), starting_pos=(25, 25), frames_no=500, save_path='test.mp4'):
    pos = video(starting_pos[0], starting_pos[1], frames_no, r, frames_res[0], size_bolita, method, corr) # Creates generated coordinates
    pos_fill = create_video(pos, frames_res[0], size_bolita) # Fills the frames with white pixels where the white ball is supposed to be.
    return create_grayscale_video(pos_fill, save_path) # Creates video, saves it on given path, it also returns the frames in a list


'''
Example for creating a video using the mice's coordinates:
'''

df = pd.read_csv('Animal 08 day 1.csv') # Read it as a pandas dataframe.
df = df.dropna() # Drop NaN values

# To use the coordinates, we need them to be integers, so we round them to the nearest integer.
df['Centre position X'] = df['Centre position X'].round().astype(int)
df['Centre position Y'] = df['Centre position Y'].round().astype(int)

# We keep the only two columns that are needed.
df = df[['Centre position X', 'Centre position Y']]

# Because of the weird origin position you mentioned in the email, we substract 315 and 80 to all registries (x - 315; y-80).
# This way we start at 0. But since there are values that are smaller than 315 and 80, We use .min() to make sure we never get a negative value.
df['Centre position X'] -= df['Centre position X'].min()
df['Centre position Y'] -= df['Centre position Y'].min()

# We transform the data from the dataframe into a list of lists, where each list contains a coordinate (Necessary because of the way I designed the function).
coordinate_list = [row.tolist() for _, row in df.iterrows()]

# We keep the first 1000 coordinates, you can change this number to be all of the coordinates, but my computer doesn't have enough RAM.
# If you want to check a certain portion of the video, you could do it like this: coordinate_list[15000:16000]
coordinate_list = coordinate_list[:1000]

# We feed into the function the coordinates, the frame resolution and the radius of the white ball (mice/rat).
# For the frame resolution to be accurate, I recommend you check the maximum value of the x and y coordinates from the original dataframe,
# And use the maximum between those two, remember the code is designed for a squared resolution.

overall_max = np.array([df['Centre position X'].max(), df['Centre position Y'].max()]).max()

pos = create_video(coordinate_list, overall_max, (3,3))
create_grayscale_video(pos, 'test.mp4')

# Finally if you want the mice to move faster or slower, you need to modify the fps in the create_grayscale_video() function.
# The more fps, the faster it moves. Also I added documentation to all functions, in case you want to modify them.