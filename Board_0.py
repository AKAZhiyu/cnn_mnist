import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw, ImageTk
from pic2array import recognize  # 确保导入了您的识别函数
import math

def save_canvas():
    filename = "to_recognize.png"
    get_canvas_image().save(filename)
    display_result(filename)

def get_canvas_image():
    x = canvas.winfo_x()
    y = canvas.winfo_y()
    width = canvas.winfo_width()
    height = canvas.winfo_height()
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    # 将Canvas上的内容绘制到图像上
    for item in canvas.find_all():
        item_coords = canvas.coords(item)
        draw.ellipse(item_coords, fill='black')
    return image

def upload_and_recognize():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        display_result(file_path)

def display_result(image_path):
    result = recognize(image_path)
    result_label.config(text="识别结果：" + str(result))

def clear_canvas():
    canvas.delete("all")

def draw(event):
    # 绘制一个小圆点以模拟连续的线条
    r = 4  # 半径
    x, y = event.x, event.y
    canvas.create_oval(x-r, y-r, x+r, y+r, fill='black', outline='black')

def animate_background():
    global hue
    hue += 0.005
    r = int(255 * (1 + math.sin(hue)) // 2)
    g = int(255 * (1 + math.sin(hue + 2)) // 2)
    b = 128  # Keeping blue constant to stay within blue-green spectrum
    color = f'#{r:02x}{g:02x}{b:02x}'
    root.config(bg=color)
    for widget in [title_label, subtitle_label, result_label, button_frame]:
        widget.config(bg=color)
    root.after(50, animate_background)

root = tk.Tk()
root.title("手写电话号码识别系统")
root.geometry("1000x650")  # Adjusted root size for a longer canvas
root.resizable(False, False)

hue = 0

font_style = ("Arial", 12, "bold")
font_title = ("Arial", 24, "bold")

title_label = tk.Label(root, text="手写电话号码识别系统", font=font_title)
title_label.pack(pady=(10, 0))

subtitle_label = tk.Label(root, text="多数字识别 - 测试准确度 99.62%", font=("Arial", 16))
subtitle_label.pack(pady=(0, 10))

canvas = tk.Canvas(root, width=800, height=400, bg='white')  # Increased width for more space to write numbers
canvas.pack(pady=10)
canvas.bind("<B1-Motion>", draw)

button_frame = tk.Frame(root)
button_frame.pack(pady=20)

button_color = '#f0f0f0'
button_style = {'font': font_style, 'bg': button_color, 'fg': 'black', 'relief': tk.GROOVE, 'padx': 5, 'pady': 5}

save_btn = tk.Button(button_frame, text="保存并识别", command=save_canvas, **button_style)
save_btn.pack(side=tk.LEFT, padx=10)

upload_btn = tk.Button(button_frame, text="上传并识别", command=upload_and_recognize, **button_style)
upload_btn.pack(side=tk.LEFT, padx=10)

clear_btn = tk.Button(button_frame, text="清除画板", command=clear_canvas, **button_style)
clear_btn.pack(side=tk.LEFT, padx=10)

result_label = tk.Label(root, text="识别结果会显示在这里", font=font_style, bg='lightgrey')
result_label.pack(pady=10)

animate_background()

root.mainloop()
