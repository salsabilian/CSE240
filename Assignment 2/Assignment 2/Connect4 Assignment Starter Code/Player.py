import numpy as np

# referenced during the assignment: https://medium.com/analytics-vidhya/artificial-intelligence-at-play-connect-four-minimax-algorithm-explained-3b5fc32e4a4f

class AIPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}:ai'.format(player_number)

    def evaluate_window(self, window):  # our scoring system for nodes placed
        score = 0
        piece = self.player_number
        if self.player_number == 1:
            opp_piece = 2
        else:
            opp_piece = 1
        if window.count(piece) == 4:
            score += 1000
        elif window.count(piece) == 3 and window.count(0) == 1:
            score += 50
        elif window.count(piece) == 2 and window.count(0) == 2:
            score += 20
        if window.count(opp_piece) == 3 and window.count(0) == 1:
            score -= 40
        return score

    @staticmethod
    def valid_placements(board):  # finds all empty spaces in the board
        valid = []
        for row in range(board.shape[0]):
            for col in range(board.shape[1]):
                if board[row, col] == 0:
                    valid.append([row, col])
        return valid

    def get_val_prob(self, board, depth, piece, opp_piece, high):
        valid = self.valid_placements(board)  # get all valid board placements
        if depth >= 3 or not valid:  # stop at depth of 3
            return self.evaluation_function(board)
        if high == 1:  # the case for getting max value
            maxExpect = -np.inf
            for row, column in valid:  # check all valid board placements
                board[row][column] = piece  # find the prob for this board placement
                current_max = self.get_val_prob(board, depth + 1, piece, opp_piece, 0)  # get the prob
                maxExpect = max(maxExpect, current_max)  # find the best prob
            return maxExpect # return that move to play it
        else:
            expectival = 0
            for row, column in valid:
                board[row][column] = opp_piece
                current_min = self.get_val_prob(board, depth + 1, piece, opp_piece, 1)  # find all possible node paths
                expectival += current_min  # add all possible node paths together

            prob = 1 / len(valid) # get how many empty spots are left on the board based on this path
            return expectival * prob # multiply our best possible paths by how many spots will be available after taking
        # that path
    def get_val(self, board, alpha, beta, depth, piece, opp_piece, high):
        valid = self.valid_placements(board)  # get all valid board placements
        if depth >= 3 or not valid:  # stop at depth of 3
            return self.evaluation_function(board)
        if high == 1:  # the case for getting max value
            for row, column in valid:  # check all valid board placements
                board[row][column] = piece  # find the current alpha for the board
                current_max = self.get_val(board, alpha, beta, depth + 1, piece, opp_piece,
                                           0)  # find the min of the max
                alpha = max(alpha, current_max)  # find the max alpha
                board[row][column] = 0
                if alpha >= beta:  # prune values that will not change the alpha
                    return alpha
            return alpha
        else:
            for row, column in valid:
                board[row][column] = opp_piece
                current_min = self.get_val(board, alpha, beta, depth + 1, piece, opp_piece, 1)  # find the max of the
                # mins
                beta = min(beta, current_min)  # find the current beta for the board
                board[row][column] = 0
                if alpha >= beta:  # prune values that will not change the beta
                    return beta
            return beta

    def get_alpha_beta_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the alpha-beta pruning algorithm

        This will play against either itself or a human player

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        values = []
        alpha = -np.inf
        beta = np.inf
        depth = 0
        if self.player_number == 1:
            opp_piece = 2
        else:
            opp_piece = 1
        for row, column in self.valid_placements(board):
            board[row][column] = self.player_number
            alpha = max(alpha, self.get_val(board, alpha, beta, depth + 1, self.player_number, opp_piece,
                                            0))  # pick the max of the min
            values.append([alpha, column])  # get alpha values for each of the columns
            board[row][column] = 0  # wipe all values after placing the value
        output = max(values, key=lambda x: x[0])  # find the max of the alpha values
        return output[1]  # place a value in the column with the highest alpha values

    def get_expectimax_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the expectimax algorithm.

        This will play against the random player, who chooses any valid move
        with equal probability

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        #  starts the same as alphabeta
        values = []
        alpha = -np.inf
        depth = 0
        if self.player_number == 1:
            opp_piece = 2
        else:
            opp_piece = 1
        for row, column in self.valid_placements(board):
            board[row][column] = self.player_number
            alpha = max(alpha, self.get_val_prob(board, depth + 1, self.player_number, opp_piece, 0)) #  call get_val_prob instead
            values.append([alpha, column])  # get alpha values for each of the columns
            board[row][column] = 0  # wipe all values after placing the value
        output = max(values, key=lambda x: x[0])  # find the max of the alpha values
        return output[1]  # place a value in the column with the highest alpha values

    def evaluation_function(self, board):
        """
        Given the current stat of the board, return the scalar value that 
        represents the evaluation function for the current player
       
        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The utility value for the current board
        """
        score = 0

        # try to control the center of the board as it maximizes connect 4 options
        center_array = [int(i) for i in list(board[:, 7 // 2])]
        center_count = center_array.count(self.player_number)
        score += center_count * 30

        # score how close we are to getting connect 4 horizontally
        for r in range(6):
            row_array = [int(i) for i in list(board[r, :])]
            for c in range(4):
                window = row_array[c: c + 4]
                score += self.evaluate_window(window)

        # score how close we are to getting a connect 4 vertically
        for c in range(7):
            col_array = [int(i) for i in list(board[:, c])]
            for r in range(3):
                window = col_array[r: r + 4]
                score += self.evaluate_window(window)

        # score how close we are to getting a connect 4 diagonally
        for r in range(3):
            for c in range(4):
                window = [board[r + i][c + i] for i in range(4)]
                score += self.evaluate_window(window)

        for r in range(3):
            for c in range(4):
                window = [board[r + 3 - i][c + i] for i in range(4)]
                score += self.evaluate_window(window)

        return score


class RandomPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'random'
        self.player_string = 'Player {}:random'.format(player_number)

    @staticmethod
    def get_move(board):
        """
        Given the current board state select a random column from the available
        valid moves.

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        valid_cols = []
        for col in range(board.shape[1]):
            if 0 in board[:, col]:
                valid_cols.append(col)

        return np.random.choice(valid_cols)


class HumanPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'human'
        self.player_string = 'Player {}:human'.format(player_number)

    @staticmethod
    def get_move(board):
        """
        Given the current board state returns the human input for next move

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """

        valid_cols = []
        for i, col in enumerate(board.T):
            if 0 in col:
                valid_cols.append(i)

        move = int(input('Enter your move: '))

        while move not in valid_cols:
            print('Column full, choose from:{}'.format(valid_cols))
            move = int(input('Enter your move: '))

        return move
