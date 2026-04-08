import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import main as app

def create_gui():
    root = tk.Tk()
    root.title("SO-101 Gesture Control")
    root.geometry("450x380")
    root.resizable(True, True)
    
    # Modern Dark Theme Colors
    BG_COLOR = "#1E1E1E"
    PANEL_BG = "#2D2D30"
    TEXT_COLOR = "#FFFFFF"
    ACCENT_COLOR = "#007ACC"
    ACCENT_HOVER = "#005C99"
    STOP_COLOR = "#E81123"
    STOP_HOVER = "#F1707A"
    
    root.configure(bg=BG_COLOR)

    style = ttk.Style()
    if 'clam' in style.theme_names():
        style.theme_use('clam')
        
    main_font = ("sans-serif", 11)
    title_font = ("sans-serif", 18, "bold")
    sub_font = ("sans-serif", 10)
    label_font = ("sans-serif", 9, "bold")
        
    style.configure("TFrame", background=BG_COLOR)
    style.configure("Panel.TFrame", background=PANEL_BG)
    style.configure("TLabel", background=BG_COLOR, foreground=TEXT_COLOR, font=main_font)
    style.configure("Header.TLabel", font=title_font, padding=(0, 10))
    style.configure("Sub.TLabel", font=sub_font, foreground="#AAAAAA")
    style.configure("TCheckbutton", background=PANEL_BG, foreground=TEXT_COLOR, font=main_font,
                    indicatorcolor=BG_COLOR, padding=5)
    style.map("TCheckbutton", background=[('active', PANEL_BG)], indicatorcolor=[('selected', ACCENT_COLOR)])
    
    # --- CONFIG FRAME ---
    config_frame = ttk.Frame(root, padding="30 30 30 30")
    
    title_label = ttk.Label(config_frame, text="SO-101 Controller", style="Header.TLabel")
    title_label.pack(anchor=tk.W)
    sub_label = ttk.Label(config_frame, text="Configure your gesture environment", style="Sub.TLabel")
    sub_label.pack(anchor=tk.W, pady=(0, 25))

    system_var = tk.StringVar(value="System 3: 3D Pose")
    mirror_var = tk.BooleanVar(value=True)

    panel = ttk.Frame(config_frame, style="Panel.TFrame")
    panel.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
    inner_panel = tk.Frame(panel, bg=PANEL_BG, padx=20, pady=20)
    inner_panel.pack(fill=tk.BOTH, expand=True)

    sys_label = tk.Label(inner_panel, text="CONTROL SYSTEM", font=label_font, 
                         fg="#888888", bg=PANEL_BG, anchor="w")
    sys_label.pack(fill=tk.X, pady=(0, 10))

    systems = [
        "System 1: Fingertips (Legacy)",
        "System 2: 2D Hand Pose (Stable)",
        "System 3: 3D Pose"
    ]
    system_map = {
        "System 1: Fingertips (Legacy)": 1,
        "System 2: 2D Hand Pose (Stable)": 2,
        "System 3: 3D Pose": 3
    }

    sys_combo = ttk.Combobox(inner_panel, textvariable=system_var, values=systems, state="readonly", font=main_font)
    sys_combo.pack(fill=tk.X, pady=2)

    tk.Frame(inner_panel, bg="#444444", height=1).pack(fill=tk.X, pady=15)

    vid_label = tk.Label(inner_panel, text="VIDEO SETTINGS", font=label_font, 
                         fg="#888888", bg=PANEL_BG, anchor="w")
    vid_label.pack(fill=tk.X, pady=(0, 10))

    chk = ttk.Checkbutton(inner_panel, text="Mirror Camera Output", variable=mirror_var)
    chk.pack(anchor=tk.W, fill=tk.X)

    # --- VIDEO FRAME ---
    video_frame = tk.Frame(root, bg=BG_COLOR)
    
    canvas_label = tk.Label(video_frame, bg="black")
    canvas_label.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

    # Control variables for the loop
    stop_flag = [False]

    def on_stop_click():
        stop_flag[0] = True

    stop_btn = tk.Button(
        video_frame, 
        text="STOP DETECTION", 
        font=("sans-serif", 10, "bold"),
        bg=STOP_COLOR, fg="white",
        activebackground=STOP_HOVER, activeforeground="white",
        relief="flat", cursor="hand2",
        command=on_stop_click
    )
    stop_btn.pack(fill=tk.X, side=tk.BOTTOM, ipady=8, padx=10, pady=(0, 10))

    def render_frame(frame):
        # Convert frame from OpenCV BGR to Pillow RGB
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_image)
        
        # Scale image down if it is too massive, to preserve UI visibility
        width, height = pil_img.size
        max_height = 640
        if height > max_height:
            ratio = max_height / height
            pil_img = pil_img.resize((int(width * ratio), max_height), Image.Resampling.LANCZOS)
        
        # Display image via ImageTk
        imgtk = ImageTk.PhotoImage(image=pil_img)
        canvas_label.imgtk = imgtk
        canvas_label.configure(image=imgtk)
        
        # Force Tkinter to process UI events (like clicking STOP) without freezing
        root.update()
        
    def check_exit():
        return stop_flag[0]

    def on_start_click():
        config_frame.pack_forget()
        
        # Let the window auto-resize to perfectly fit the camera stream and stop button
        root.geometry("")
        video_frame.pack(fill=tk.BOTH, expand=True)
        root.update()

        # Reset flag
        stop_flag[0] = False
        
        sys_val = system_map[system_var.get()]
        mirror_val = mirror_var.get()

        try:
            # Start loop in main thread; render_frame will call root.update()
            app.main(control_system=sys_val, mirror_video=mirror_val, 
                     frame_callback=render_frame, check_exit_callback=check_exit)
        except Exception as e:
            print(f"Error during detection: {e}")
        finally:
            video_frame.pack_forget()
            root.geometry("450x380")
            config_frame.pack(fill=tk.BOTH, expand=True)

    start_btn = tk.Button(
        config_frame, 
        text="START DETECTION", 
        font=("sans-serif", 10, "bold"),
        bg=ACCENT_COLOR, fg=TEXT_COLOR,
        activebackground=ACCENT_HOVER, activeforeground=TEXT_COLOR,
        relief="flat", cursor="hand2",
        command=on_start_click
    )
    start_btn.pack(fill=tk.X, side=tk.BOTTOM, ipady=8)
    
    config_frame.pack(fill=tk.BOTH, expand=True)

    root.mainloop()

if __name__ == "__main__":
    create_gui()
