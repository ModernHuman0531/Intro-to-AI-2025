import numpy as np
import random
import game

def print_INFO():
    """
    Prints your homework submission details.
    Please replace the placeholders (date, name, student ID) with valid information
    before submitting.
    """
    print(
        """========================================
        DATE: 2025/04/04
        STUDENT NAME: WEI YU-SHYANG
        STUDENT ID: 110612025
        ========================================
        """)

""" Important functions
1. grid.terminate(): Using this to stop game when game reaches win, loss or draw.
2. get_heuristic(grid): Using this to evaluate the board when the maximum depth is reached.
3. game.drop_piece(grid, col): Using this to simulate the move.
"""

#
# Basic search functions: Minimax and Alpha‑Beta
#

# Depth 4 means that the search will look 4 moves ahead.
def minimax(grid, depth, maximizingPlayer, dep=4):
    """
    TODO (Part 1): Implement recursive Minimax search for Connect Four.

    Return:
      (boardValue, {setOfCandidateMoves})

    Where:
      - boardValue is the evaluated utility of the board state
      - {setOfCandidateMoves} is a set of columns that achieve this boardValue
    """
    # Placeholder return to keep function structure intact
    """
    1. Use grid.terminate() to check the end state(Win, Loss or Draw)
    2. Initialize the best value to -inf or inf based on maximizingPlayer.
    3. Loop through all the valid moves by using grid.valid, for each move, 
       simulate the move using grid.drop_piece(col). 
    4. Call minimax recursively with the new grid, depth-1, and opposite player
    5. Update the maximum layer with the the value that higher than the best value.
    6. Update the minimum layer with the value that lower than the best value.
    7. If the depth is 0, return the heuristic value using get_heuristic(grid).
    """
    # Return the value when the game is over or the depth is 0.
    if grid.terminate() or depth == 0:
        # If the game is over, return the heuristic value and the set of valid move
        # when the premove step is reached, calculate the current heuristic value.
        return get_heuristic(grid), set(grid.valid)
    
    # Initialize the best value based on maximizingPlayer.
    # For the player we always want to maximize the value.
    # For the opponent we always want to minimize the value.
    if maximizingPlayer:
        bestValue = -np.inf
    else:
        bestValue = np.inf
    # Every layer have their own set for the candidate of the best move.
    best_move = set()

    # Loop through all the valid moves
    for col in grid.valid:
        new_grid = game.drop_piece(grid, col) # Simulate the move
        value, _ = minimax(new_grid, depth-1, not maximizingPlayer) # Call minimax recurisively with the new grid, depth-1 and opposite player
    # _: We don't need the set of candidate moves here, ignore that return value.

        #Update the maximum layer with the the value that higher than the best value.
        if maximizingPlayer:
            # If the value is the best value, update the best_value and that move is the only best move in set.
            if value > bestValue:
                bestValue = value
                best_move = {col}
                
            # If the value is equal to the best value, add the move to the set of the best move candidate.
            elif value == bestValue:
                best_move.add(col)

        # Update the minimum layer (the opponent side) with the value that lower than the best value.
        else:
            # If the value is smaller than the best value, update the best_value and create the new set of the best_move.
            if value < bestValue:
                bestValue = value
                best_move = {col}
            # If the value is equal to the best value, add the move to the set of the best move candidate.
            elif value == bestValue:
                best_move.add(col)
    return bestValue, best_move



def alphabeta(grid, depth, maximizingPlayer, alpha, beta, dep=2):
    """
    TODO (Part 2): Implement Alpha-Beta pruning as an optimization to Minimax.

    Return:
      (boardValue, {setOfCandidateMoves})

    Where:
      - boardValue is the evaluated utility of the board state
      - {setOfCandidateMoves} is a set of columns that achieve this boardValue
      - Prune branches when alpha >= beta
    """
    # Placeholder return to keep function structure intact
    # Alpha is the maximum lower bound of possible values, Beta is the minimum upper bound of possible values.

    """
    1. Use grid.terminate() to check the end state(Win, Loss or Draw)
    2.Initialize the best value to -inf or inf based on maximizingPlayer.
    3. Loop through all the valid moves by using grid.valid for each move.
    4. Call the alphabeta recursively with new grid, depth-1, and opposite player.
    5. When encounter the maximum layer, choose the maximum value between best value and alpha.
    6. When encounter the minimum layer, choose the minimum value between best value and beta.
    7. If alpha >= beta, prune the branch and break the loop.
    8. If the depth is 0, return the heuristic value using get_heuristic(grid).
    """
    if grid.terminate() or depth == 0:
        return get_heuristic(grid), set(grid.valid)
    # Initialize the best value based on maximizingPlayer.
    if maximizingPlayer:
        bestValue = -np.inf
    else:
        bestValue = np.inf
    best_move = set()

    # Loop through all the valid moves
    for col in grid.valid:
        new_grid = game.drop_piece(grid, col) # Simulate the move
        value, _ = alphabeta(new_grid, depth-1, not maximizingPlayer, alpha, beta)
        # _: We don't need the set of candidate moves here, ignore that return value.

        # Update the maximum layer with the the value that higher than the best value.
        if maximizingPlayer:
            if value > bestValue:
                bestValue = value
                best_move = {col}# If best_value > value, then col is the only best move in set.
            elif value == bestValue:
                best_move.add(col)
            
            # Update alpha with the maximum value between bestValue and alpha.
            alpha = max(alpha, bestValue)
            # If alpha >= beta, prune the branch and break the loop.
            if alpha >= beta:
                break
        # Update the minimum layer (the opponent side) with the value that lower than the best value.
        else:
            if value < bestValue:
                bestValue = value
                best_move = {col}# If best_value < value, then col is the only best move in set.
            elif value == bestValue:
                best_move.add(col)
            # Update beta with the minimum value between bestValue and beta.
            beta = min(beta, bestValue)

            # If alpha >= beta, prune the branch and break the loop.
            if alpha >= beta:
                break
    # Return the best value and the set of candidate moves
    return bestValue, best_move


#
# Basic agents
#

def agent_minimax(grid):
    """
    Agent that uses the minimax() function with a default search depth (e.g., 4).
    Must return a single column (integer) where the piece is dropped.
    """
    return random.choice(list(minimax(grid, 4, True)[1]))


def agent_alphabeta(grid):
    """
    Agent that uses the alphabeta() function with a default search depth (e.g., 4).
    Must return a single column (integer) where the piece is dropped.
    """
    return random.choice(list(alphabeta(grid, 4, True, -np.inf, np.inf)[1]))


def agent_reflex(grid):
    """
    A simple reflex agent provided as a baseline:
      - Checks if there's an immediate winning move.
      - Otherwise picks a random valid column.
    """
    wins = [c for c in grid.valid if game.check_winning_move(grid, c, grid.mark)]
    if wins:
        return random.choice(wins)
    return random.choice(grid.valid)


def agent_strong(grid):
    """
    TODO (Part 3): Design your own agent (depth = 4) to consistently beat the Alpha-Beta agent (depth = 4).
    This agent will typically act as Player 2.
    """
    # Placeholder logic that calls your_function().
    return random.choice(list(your_function(grid, 4, False, -np.inf, np.inf)[1]))


#
# Heuristic functions
#

def get_heuristic(board):
    """
    Evaluates the board from Player 1's perspective using a basic heuristic.

    Returns:
      - Large positive value if Player 1 is winning
      - Large negative value if Player 2 is winning
      - Intermediate scores based on partial connect patterns
    """
    num_twos       = game.count_windows(board, 2, 1)
    num_threes     = game.count_windows(board, 3, 1)
    num_twos_opp   = game.count_windows(board, 2, 2)
    num_threes_opp = game.count_windows(board, 3, 2)

    score = (
          1e10 * board.win(1)
        + 1e6  * num_threes
        + 10   * num_twos
        - 10   * num_twos_opp
        - 1e6  * num_threes_opp
        - 1e10 * board.win(2)
    )
    return score


def get_heuristic_strong(board):
    """
    TODO (Part 3): Implement a more advanced board evaluation for agent_strong.
    Currently a placeholder that returns 0.
    """

    """ Key points to consider:
    1. Favor center control for better future move flexibility: Add weight to center columns.
    2. Identify moves leading to a guaranteed win and prioritize them: Detect all the valid moves and check if they lead immediately to a win, give it highest positive weight.
    3. Recognize imminent opponent victories and block them: Detect all the valid moves and check if they lead to a win for the opponent, give it highest negative weight.
    4. Prioritize moves that provide multiple winning options: Potential of creating mulitiple connect of num_threes
    """
    # Basic count of connections for Player 1 and Player 2.
    num_twos = game.count_windows(board, 2, 1)
    num_threes = game.count_windows(board, 3, 1)
    num_twos_opp = game.count_windows(board, 2, 2)
    num_threes_oop = game.count_windows(board, 3, 2)

    # Central control: Count how many pieces are in the center buttom row.
    central_column = [2,3,4]
    center_control = sum(1 for move in central_column if board.table[board.row-1][move] == 1)

    # Find potential winning move for player 1 in 1 move.
    winning_moves = sum([1 for move in board.valid if game.check_winning_move(board, move, 1)])

    # Find potential winning move for player 2 in 1 move.
    opp_winning_moves = sum([1 for move in board.valid if game.check_winning_move(board, move, 2)])

    # Calculate the potential of creating multiple connect of num_threes.
    double_threats = 0
    for c in board.valid:
        new_grid = game.drop_piece(board, c)
        if game.count_windows(new_grid, 3, 1) >= 2:
            double_threats += 1

    # Calculate the potential of creating multiple connect of num_threes for player 2.
    double_threats_opp = 0
    for c in board.valid:
        new_grid = game.drop_piece(board, c)
        if game.count_windows(new_grid, 3, 2) >= 2:
            double_threats_opp += 1

    # Create a heuristic score based on the above factors.
    score = (
        1e10 * board.win(1) # Direct win for player 1
        + 1e8 * winning_moves # Next move is a winning move for player 1
        + 1e6 * num_threes # Number of threes for player 1
        + 5e4 * double_threats # Potential to create multiple connect of num_threes
        + 1e3 * center_control # Center control
        + 10 * num_twos # Number of twos for player 1
        - 10 * num_twos_opp # Number of twos for player 2
        - 5e4 * double_threats_opp # Potential to create multiple connect of num_threes for player 2
        - 1e6 * num_threes_oop # Number of threes for player 2
        - 1e8 * opp_winning_moves # Next move is a winning move for player 2
        - 1e10 * board.win(2) # Direct win for player 2
    )
    return score  


def your_function(grid, depth, maximizingPlayer, alpha, beta, dep=4):
    """
    A stronger search function that uses get_heuristic_strong() instead of get_heuristic().
    You can employ advanced features (e.g., improved move ordering, deeper lookahead).

    Return:
      (boardValue, {setOfCandidateMoves})

    Currently a placeholder returning (0, {0}).
    """
    """Optimization the alphabeta function by using get_heuristic_strong() instead of get_heuristic().
    1. Check if there is immediate win for player 1 or palyer 2, if so just randomly choose one of the winning move.
    2. Give the order of the move based on the heuristic value, 
        explore the highest value first if your function is maximizing player,
        explore the lowest value first if your function is minimizing player.
    3.  Still apply the alpha-beta pruning to prune the branches.

    """
    # If thre is immediate win for player 1 or player 2, if so just randomly choose one of the winning move.
    if maximizingPlayer:
        winning_moves = [c for c in grid.valid if game.check_winning_move(grid, c, 1)]
        if winning_moves:
            return float('inf'), set(winning_moves)
    else:
        winning_moves = [c for c in grid.valid if game.check_winning_move(grid, c, 2)]
        if winning_moves:
            return float('-inf'), set(winning_moves)
        
    # Check if the game is over or the depth is 0.
    if grid.terminate() or depth == 0:
        return get_heuristic_strong(grid), set(grid.valid)
    
    # Initialize the best value based on maximizingPlayer.
    if maximizingPlayer:
        bestValue = -np.inf
    else:
        bestValue = np.inf
    best_move = set()

    # Set the order of the move based on the heuristic value to explore the tree.
    move_score = [] # Store the data in the form of (col, score)
    for col in grid.valid:
        new_grid = game.drop_piece(grid, col) # Simulate the move
        score = get_heuristic_strong(new_grid) # Get the heuristic value of the new grid
        move_score.append((score, col)) # Append the score and the column to the list

    # Based on the maximizingPlayer, sort the move_score list in descending or ascending order.
    move_score.sort(key = lambda x: -x[0] if maximizingPlayer else x[0])

    # Loop through all the valid moves in the order of the heuristic value.
    for _, col in move_score:
        new_grid = game.drop_piece(grid, col) # Simulate the move
        value, _ = your_function(new_grid, depth-1, not maximizingPlayer, alpha, beta)
        # _: We don't need the set of candidate moves here, ignore that return value.

        # Update the maximum layer with the the value that higher than the best value.
        if maximizingPlayer:
            if value > bestValue:
                bestValue = value
                best_move = {col}
            elif value == bestValue:
                best_move.add(col)

            # Update alpha with the maximum value between bestValue and alpha.
            alpha = max(alpha, bestValue)
            # If alpha >= beta, prune the branch and break the loop.
            if alpha >= beta:
                break
        # Update the minimum layer (the opponent side) with the value that lower than the best value.
        else:
            if value < bestValue:
                bestValue = value
                best_move = {col}
            elif value == bestValue:
                best_move.add(col)

            # Update beta with the minimum value between bestValue and beta.
            beta = min(beta, bestValue)
            # If alpha >= beta, prune the branch and break the loop.
            if alpha >= beta:
                break
    # Return the best value and the set of candidate moves
    return bestValue, best_move
