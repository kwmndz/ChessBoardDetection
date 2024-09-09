import tkinter as tk
import pyautogui
from PIL import ImageGrab
from getBoardFEN import getSquares, getFen

def select_area():
    # Create a fullscreen transparent window to capture mouse events
    root = tk.Tk()
    root.attributes("-alpha", 0.3)
    root.attributes("-fullscreen", True)

    # Create a canvas to draw the selection rectangle
    canvas = tk.Canvas(root, bg="black", highlightthickness=0)
    canvas.pack(fill=tk.BOTH, expand=True)

    # Variables to store the coordinates of the selection rectangle
    x1, y1, x2, y2 = 0, 0, 0, 0
    
    label.config(text="Select the chessboard area")

    def on_mouse_press(event):
        nonlocal x1, y1
        x1, y1 = event.x, event.y

    def on_mouse_release(event):
        nonlocal x2, y2
        x2, y2 = event.x, event.y

        # Enforce a square shape for the selection rectangle
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        size = min(width, height)
        if x2 < x1:
            x2 = x1 - size
        else:
            x2 = x1 + size
        if y2 < y1:
            y2 = y1 - size
        else:
            y2 = y1 + size

        # Destroy the transparent window after the selection is made
        root.destroy()

        label.config(text="Processing...")

        # Take a screenshot of the selected portion
        screenshot = ImageGrab.grab(bbox=(x1, y1, x2, y2))
        screenshot.save('temp.jpg')

        # Get the squares from the screenshot
        squares = getSquares('temp.jpg')
        if not squares:
            label.config(text="No chessboard detected")
            return
        
        fen = getFen('model.pth', squares)

        # Display the next best move
        label.config(text=f"Next Best Move: {fen}")

    def on_mouse_move(event):
        nonlocal x2, y2
        x2, y2 = event.x, event.y

        # Enforce a square shape for the selection rectangle
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        size = min(width, height)
        if x2 < x1:
            x2 = x1 - size
        else:
            x2 = x1 + size
        if y2 < y1:
            y2 = y1 - size
        else:
            y2 = y1 + size

        # Clear the previous selection rectangle
        canvas.delete("selection")

        # Draw the new selection rectangle with a thicker outline
        canvas.create_rectangle(x1, y1, x2, y2, outline="red", fill="red", stipple="gray50", width=4, tags="selection")

    # Bind mouse events to the canvas
    canvas.bind("<ButtonPress-1>", on_mouse_press)
    canvas.bind("<B1-Motion>", on_mouse_move)
    canvas.bind("<ButtonRelease-1>", on_mouse_release)

    # Start the main loop for the transparent window
    root.mainloop()

# Create the main window
window = tk.Tk()
window.title("Chess Solver")

# Set the window size
window.geometry("400x200")

# Create a button to take a screenshot
button = tk.Button(window, text="Take Screenshot", command=select_area)
button.pack()

# Create a label to display the next best move
label = tk.Label(window, text="Next Best Move: ")
label.pack()

# Start the main loop
window.mainloop()