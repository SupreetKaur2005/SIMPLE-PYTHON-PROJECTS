# tic_tac_toe_frontend.py
import tkinter as tk
from tkinter import messagebox
from TicTacToe import check_winner, check_draw, best_move

# Initialize the main window
root = tk.Tk()
root.title("Tic-Tac-Toe")

# Initialize the board
board = [[' ' for _ in range(3)] for _ in range(3)]
current_player = 'X'

# Function to handle button click
def button_click(row, col):
    global current_player
    if board[row][col] == ' ':
        board[row][col] = current_player
        buttons[row][col].config(text=current_player)
        winner = check_winner(board)
        if winner:
            messagebox.showinfo("Tic-Tac-Toe", f"Player {winner} wins!")
            reset_board()
        elif check_draw(board):
            messagebox.showinfo("Tic-Tac-Toe", "It's a draw!")
            reset_board()
        else:
            current_player = 'O' if current_player == 'X' else 'X'
            if current_player == 'O':
                ai_move()

# Function for AI move
def ai_move():
    move = best_move(board)
    if move:
        board[move[0]][move[1]] = 'O'
        buttons[move[0]][move[1]].config(text='O')
        winner = check_winner(board)
        if winner:
            messagebox.showinfo("Tic-Tac-Toe", f"Player {winner} wins!")
            reset_board()
        elif check_draw(board):
            messagebox.showinfo("Tic-Tac-Toe", "It's a draw!")
            reset_board()
        else:
            global current_player
            current_player = 'X'

# Function to reset the board
def reset_board():
    global board, current_player
    board = [[' ' for _ in range(3)] for _ in range(3)]
    current_player = 'X'
    for row in range(3):
        for col in range(3):
            buttons[row][col].config(text=' ')

# Create buttons for the board
buttons = [[None for _ in range(3)] for _ in range(3)]
for row in range(3):
    for col in range(3):
        buttons[row][col] = tk.Button(root, text=' ', font=('normal', 40), width=5, height=2,
                                      command=lambda row=row, col=col: button_click(row, col))
        buttons[row][col].grid(row=row, column=col)

# Start the main loop
root.mainloop()
