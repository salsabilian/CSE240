import numpy as np
import helper
import random

#   This class has all the functions and variables necessary to implement snake game
#   We will be using Q learning to do this

class SnakeAgent:

    #   This is the constructor for the SnakeAgent class
    #   It initializes the actions that can be made,
    #   Ne which is a parameter helpful to perform exploration before deciding next action,
    #   LPC which ia parameter helpful in calculating learning rate (lr)
    #   gamma which is another parameter helpful in calculating next move, in other words
    #            gamma is used to blalance immediate and future reward
    #   Q is the q-table used in Q-learning
    #   N is the next state used to explore possible moves and decide the best one before updating
    #           the q-table
    def __init__(self, actions, Ne, LPC, gamma):
        self.actions = actions
        self.Ne = Ne
        self.LPC = LPC
        self.gamma = gamma
        self.reset()

        # Create the Q and N Table to work with
        self.Q = helper.initialize_q_as_zeros()
        self.N = helper.initialize_q_as_zeros()


    #   This function sets if the program is in training mode or testing mode.
    def set_train(self):
        self._train = True

    #   This function sets if the program is in training mode or testing mode.
    def set_eval(self):
        self._train = False

    #   Calls the helper function to save the q-table after training
    def save_model(self):
        helper.save(self.Q)

    #   Calls the helper function to load the q-table when testing
    def load_model(self):
        self.Q = helper.load()

    #   resets the game state
    def reset(self):
        self.points = 0
        self.s = None
        self.a = None

    #   This is a function you should write.
    #   Function Helper:IT gets the current state, and based on the
    #   current snake head location, body and food location,
    #   determines which move(s) it can make by also using the
    #   board variables to see if its near a wall or if  the
    #   moves it can make lead it into the snake body and so on.
    #   This can return a list of variables that help you keep track of
    #   conditions mentioned above.
    def helper_func(self, state):
        # state values equivalents
        head_x = state[0]
        head_y = state[1]
        food_x = state[3]
        food_y = state[4]
        # variables use to identify current board environment specified in helper.py initialize_q_as_zeros function
        wall_x_state = 0
        wall_y_state = 0
        food_x_state = 0
        food_y_state = 0
        body = []
        for i, j in state[2]:
            body.append([i, j])
        # are we about to hit the body
        snake_body = [0, 0, 0, 0] #keeps track of if were about to hit the snake body [y-1, y+1, x-1, x+1] to hit snake
        if [head_x, head_y-1] in body:
            snake_body[0] = 1
        if [head_x, head_y+1] in body:
            snake_body[1] = 1
        if [head_x-1, head_y] in body:
            snake_body[2] = 1
        if [head_x+1, head_y] in body:
            snake_body[3] = 1
        # are we about to hit a wall
        if head_x == 40:
            wall_x_state = 1
        if head_x == 480:
            wall_x_state = 2
        if head_y == 40:
            wall_y_state = 1
        if head_y == 480:
            wall_y_state = 2
        # where's the food at
        if food_x - head_x > 0:
            food_x_state = 1
        elif food_x - head_x < 0:
            food_x_state = 2
        if food_y - head_y > 0:
            food_y_state = 1
        elif food_y - head_y < 0:
            food_y_state = 2
        return [wall_x_state, wall_y_state, food_x_state, food_y_state, snake_body[0], snake_body[1], snake_body[2], snake_body[3]]
        # YOUR CODE HERE
        # YOUR CODE HERE
        # YOUR CODE HERE
        # YOUR CODE HERE
        # YOUR CODE HERE


    # Computing the reward, need not be changed.
    def compute_reward(self, points, dead):
        if dead:
            return -1
        elif points > self.points:
            return 1
        else:
            return -0.1

    #   This is the code you need to write.
    #   This is the reinforcement learning agent
    #   use the helper_func you need to write above to
    #   decide which move is the best move that the snake needs to make
    #   using the compute reward function defined above.
    #   This function also keeps track of the fact that we are in
    #   training state or testing state so that it can decide if it needs
    #   to update the Q variable. It can use the N variable to test outcomes
    #   of possible moves it can make.
    #   the LPC variable can be used to determine the learning rate (lr), but if
    #   you're stuck on how to do this, just use a learning rate of 0.7 first,
    #   get your code to work then work on this.
    #   gamma is another useful parameter to determine the learning rate.
    #   based on the lr, reward, and gamma values you can update the q-table.
    #   If you're not in training mode, use the q-table loaded (already done)
    #   to make moves based on that.
    #   the only thing this function should return is the best action to take
    #   ie. (0 or 1 or 2 or 3) respectively.
    #   The parameters defined should be enough. If you want to describe more elaborate
    #   states as mentioned in helper_func, use the state variable to contain all that.
    def agent_action(self, state, points, dead):

        def calculate_reward(state, dead, points, previous):
            # how up down left and right are identified
            u = 0
            d = 1
            l = 2
            r = 3
            #previous = self.helper_func(self.s) # get identifiable information for the previous board state using helper function
            current = self.helper_func(state)  # get identifiable information for the current board state using the helper func
            reward = self.compute_reward(points, dead)  # calculate our current reward
            # get the current rewards for all moves up, down, left, or right (so we can pick the best one)
            up = self.Q[current[0]][current[1]][current[2]][current[3]][current[4]][current[5]][current[6]][current[7]][u]
            down = self.Q[current[0]][current[1]][current[2]][current[3]][current[4]][current[5]][current[6]][current[7]][d]
            left = self.Q[current[0]][current[1]][current[2]][current[3]][current[4]][current[5]][current[6]][current[7]][l]
            right = self.Q[current[0]][current[1]][current[2]][current[3]][current[4]][current[5]][current[6]][current[7]][r]
            # get the max move
            maxQ = max(up, down, left, right)
            #used for the bellman equation
            q = self.Q[previous[0]][previous[1]][previous[2]][previous[3]][previous[4]][previous[5]][previous[6]][previous[7]][self.a]
            fixed_LPC = self.LPC / (self.LPC + self.N[previous[0]][previous[1]][previous[2]][previous[3]][previous[4]][previous[5]][previous[6]][previous[7]][self.a])
            val = q + fixed_LPC*(reward + self.gamma*maxQ - q)  # calculate the bellman equation
            return val

        move = [0, 0, 0, 0] # we have to either go up, down, left, or right we will find the one thats the best based on Q
        if dead:
            previous = self.helper_func(self.s)  # use the helper function to identify moves for the Q_Table
            # set the value for the move that killed us using the bellman equation
            score = calculate_reward(state, dead, points, previous) # set the score to let us know not to go here again cause it killed us
            self.Q[previous[0]][previous[1]][previous[2]][previous[3]][previous[4]][previous[5]][previous[6]][previous[7]][self.a] = score
            self.reset()  # reset the game states
            return
        present = self.helper_func(state)  # get the current states moves
        if self._train and self.s is not None and self.a is not None:  # for training mode if the previous state and action are not set we will get poorly trained results
            previous = self.helper_func(self.s) # get the previous state and action move to identify for scoring
            score = calculate_reward(state, dead, points, previous)  # score it
            self.Q[previous[0]][previous[1]][previous[2]][previous[3]][previous[4]][previous[5]][previous[6]][previous[7]][self.a] = score
        for i in range(4):  # go through all the possible moves and find its Q and N value to determine if its the best option
            n = self.N[present[0]][present[1]][present[2]][present[3]][present[4]][present[5]][present[6]][present[7]][i]
            q = self.Q[present[0]][present[1]][present[2]][present[3]][present[4]][present[5]][present[6]][present[7]][i]
            if n < self.Ne:  # if the n value is less than Ne take a random value for the move (this allows us to make random moves)
                move[i] = 1
            else: # otherwise set the q value
                move[i] = q
        action = np.argmax(move)  # pick the best move
        # increment the N to indicate it's the best action for next time
        self.N[present[0]][present[1]][present[2]][present[3]][present[4]][present[5]][present[6]][present[7]][action] += 1
        self.s = state  # save the state, action, and points taken for our next time
        self.a = action
        self.points = points
        # YOUR CODE HERE
        # YOUR CODE HERE
        # YOUR CODE HERE
        # YOUR CODE HERE
        # YOUR CODE HERE
        # YOUR CODE HERE
        # YOUR CODE HERE

        #UNCOMMENT THIS TO RETURN THE REQUIRED ACTION.
        return action