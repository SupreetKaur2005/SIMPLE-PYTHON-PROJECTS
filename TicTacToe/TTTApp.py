import kivy
from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from TicTacToe import check_winner, check_draw, best_move

class TicTacToeApp(App):
    def build(self):
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.current_player = 'X'
        self.layout = GridLayout(cols=3)
        self.buttons = [[None for _ in range(3)] for _ in range(3)]
        
        for row in range(3):
            for col in range(3):
                button = Button(text=' ', font_size=40, on_press=self.button_click)
                self.layout.add_widget(button)
                self.buttons[row][col] = button
        
        return self.layout

    def button_click(self, instance):
        row, col = self.get_button_pos(instance)
        if self.board[row][col] == ' ':
            self.board[row][col] = self.current_player
            instance.text = self.current_player
            winner = check_winner(self.board)
            if winner:
                self.show_popup(f"Player {winner} wins!")
                self.reset_board()
            elif check_draw(self.board):
                self.show_popup("It's a draw!")
                self.reset_board()
            else:
                self.current_player = 'O' if self.current_player == 'X' else 'X'
                if self.current_player == 'O':
                    self.ai_move()

    def get_button_pos(self, instance):
        for row in range(3):
            for col in range(3):
                if self.buttons[row][col] == instance:
                    return row, col
        return None

    def ai_move(self):
        move = best_move(self.board)
        if move:
            row, col = move
            self.board[row][col] = 'O'
            self.buttons[row][col].text = 'O'
            winner = check_winner(self.board)
            if winner:
                self.show_popup(f"Player {winner} wins!")
                self.reset_board()
            elif check_draw(self.board):
                self.show_popup("It's a draw!")
                self.reset_board()
            else:
                self.current_player = 'X'

    def show_popup(self, message):
        popup = Popup(title='Game Over', content=Label(text=message), size_hint=(0.6, 0.4))
        popup.open()

    def reset_board(self):
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.current_player = 'X'
        for row in range(3):
            for col in range(3):
                self.buttons[row][col].text = ' '

if __name__ == '__main__':
    TicTacToeApp().run()
