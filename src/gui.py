import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import logging
import math
import random
import time
from pathlib import Path
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve

from .engine import ShuttlePredictorEngine
from .config import AppConfig
from .utils import play_sound_cross_platform, DataLoadError

class MetricCard(ctk.CTkFrame):
    """A reusable widget for displaying metrics with a title, value, and trend."""
    def __init__(self, parent, title, value, trend="neutral", color=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.configure(fg_color="#1e293b", corner_radius=12, border_width=1, border_color="#334155")
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
        # Title
        self.title_label = ctk.CTkLabel(
            self, 
            text=title.upper(), 
            font=("Roboto", 11, "bold"), 
            text_color="#94a3b8"
        )
        self.title_label.grid(row=0, column=0, sticky="w", padx=12, pady=(12, 0))
        
        # Value
        self.value_label = ctk.CTkLabel(
            self, 
            text=value, 
            font=("Roboto", 24, "bold"), 
            text_color="white"
        )
        self.value_label.grid(row=1, column=0, sticky="w", padx=12, pady=(4, 0))
        
        # Trend Indicator (Simple visual cue)
        trend_color = "#94a3b8"
        if trend == "up": trend_color = "#10b981"
        elif trend == "down": trend_color = "#ef4444"
        
        self.trend_bar = ctk.CTkProgressBar(self, height=4, progress_color=trend_color)
        self.trend_bar.set(1.0 if trend != "neutral" else 0.0)
        self.trend_bar.grid(row=2, column=0, sticky="ew", padx=12, pady=(8, 12))

class AURAKShuttlePredictor(ctk.CTk):
    """
    Main application class for the AURAK Shuttle Arrival Predictor.
    Implements a professional desktop application with tabbed interface.
    """
    def __init__(self):
        super().__init__()
        
        # --- Design System ---
        self.config = AppConfig()
        self.palette = self.config.PALETTE
        self.typography = self.config.get_fonts(ctk.CTkFont)
        # --- End Design System ---
        
        # Configure the main window
        self.title(self.config.WINDOW_TITLE)
        self.geometry(self.config.WINDOW_SIZE)
        self.minsize(*self.config.MIN_WINDOW_SIZE)
        
        # Make window resizable
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # Initialize Engine
        self.engine = ShuttlePredictorEngine()
        self.engine.add_observer(self.on_engine_update)
        
        # UI State
        self.is_simulating = False
        self.simulation_thread = None
        self.prediction_log_entries = []
        
        # Theme Colors
        self.panel_fg_color = self.palette["panel_fg"]
        self.panel_border_color = self.palette["panel_border"]
        self.section_fg_color = self.palette["section_fg"]

        # Animated gradient background configuration
        self.gradient_palettes = self.config.GRADIENT_PALETTES
        self.gradient_transition_steps = 180
        self.current_gradient_index = 0
        self.next_gradient_index = 1
        self.current_gradient_step = 0
        
        # Build UI
        self.setup_background_animation()
        self.create_interface()
        
        # Start background animation
        self.animate_background_gradient()

    def on_engine_update(self, event_type: str, data: any):
        """Handle updates from the engine."""
        if event_type == "training_complete":
            self.after(0, self._training_complete)
        elif event_type == "data_loaded":
            self.after(0, lambda: self.data_status_label.configure(text=f"Data loaded: {data} records"))
        elif event_type == "model_loaded":
             self.after(0, lambda: messagebox.showinfo("Success", f"Model loaded from {data}"))

    def create_interface(self):
        """Create the main dashboard interface."""
        self.setup_main_container()
        self.create_control_deck()
        self.create_simulation_deck()
        self.create_analytics_deck()

    def setup_background_animation(self):
        """Create animated gradient background."""
        self.bg_canvas = tk.Canvas(self, highlightthickness=0)
        self.bg_canvas.place(x=0, y=0, relwidth=1, relheight=1)
        self.bind("<Configure>", self.on_background_resize)

    def draw_background_gradient(self, start_color, end_color):
        """Draw a simple vertical gradient on the background canvas."""
        width = self.winfo_width()
        height = self.winfo_height()
        self.bg_canvas.delete("gradient")
        
        # Create a gradient using lines (efficient enough for this purpose)
        limit = height
        r1, g1, b1 = self.winfo_rgb(start_color)
        r2, g2, b2 = self.winfo_rgb(end_color)
        
        r_ratio = (r2 - r1) / limit
        g_ratio = (g2 - g1) / limit
        b_ratio = (b2 - b1) / limit

        for i in range(limit):
            nr = int(r1 + (r_ratio * i))
            ng = int(g1 + (g_ratio * i))
            nb = int(b1 + (b_ratio * i))
            color = "#%4.4x%4.4x%4.4x" % (nr, ng, nb)
            self.bg_canvas.create_line(0, i, width, i, tags=("gradient",), fill=color)
        self.bg_canvas.tag_lower("gradient")

    def interpolate_color(self, color_a, color_b, t):
        """Blend between two hex colors."""
        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

        def rgb_to_hex(rgb_color):
            return '#{:02x}{:02x}{:02x}'.format(*rgb_color)

        a = hex_to_rgb(color_a)
        b = hex_to_rgb(color_b)
        mixed = tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))
        return rgb_to_hex(mixed)

    def animate_background_gradient(self):
        """Animate background gradient transitions."""
        start_palette = self.gradient_palettes[self.current_gradient_index]
        end_palette = self.gradient_palettes[self.next_gradient_index]
        progress = self.current_gradient_step / max(self.gradient_transition_steps, 1)
        start_color = self.interpolate_color(start_palette[0], end_palette[0], progress)
        end_color = self.interpolate_color(start_palette[1], end_palette[1], progress)
        self.draw_background_gradient(start_color, end_color)

        self.current_gradient_step += 1
        if self.current_gradient_step > self.gradient_transition_steps:
            self.current_gradient_step = 0
            self.current_gradient_index = self.next_gradient_index
            self.next_gradient_index = (self.next_gradient_index + 1) % len(self.gradient_palettes)

        self.after(120, self.animate_background_gradient)

    def on_background_resize(self, _event):
        """Redraw gradient on window resize."""
        current_palette = self.gradient_palettes[self.current_gradient_index]
        next_palette = self.gradient_palettes[self.next_gradient_index]
        progress = self.current_gradient_step / max(self.gradient_transition_steps, 1)
        start_color = self.interpolate_color(current_palette[0], next_palette[0], progress)
        end_color = self.interpolate_color(current_palette[1], next_palette[1], progress)
        self.draw_background_gradient(start_color, end_color)

    def setup_main_container(self):
        """Configure main glass container and grid layout."""
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.pack(fill="both", expand=True, padx=30, pady=30)
        self.main_container.grid_rowconfigure(0, weight=3)
        self.main_container.grid_rowconfigure(1, weight=2)
        self.main_container.grid_columnconfigure(0, weight=3)  # Control Deck
        self.main_container.grid_columnconfigure(1, weight=5)  # Sim + Analytics

    def create_control_deck(self):
        """Build the left-hand control panel."""
        self.control_deck = ctk.CTkFrame(
            self.main_container,
            corner_radius=24,
            fg_color=self.panel_fg_color,
            border_width=1,
            border_color=self.panel_border_color,
        )
        self.control_deck.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 24))
        self.control_deck.grid_rowconfigure(0, weight=1)

        self.control_scroll_frame = ctk.CTkScrollableFrame(
            self.control_deck, fg_color="transparent"
        )
        self.control_scroll_frame.pack(fill="both", expand=True, padx=24, pady=24)

        self.create_theme_and_sound_controls()
        self.create_data_section()
        self.create_model_section()
        self.create_prediction_section()
        self.create_predict_button()

    def _create_glass_section(self, parent, title):
        """Create a semi-transparent section with a title."""
        section = ctk.CTkFrame(
            parent,
            fg_color=self.section_fg_color,
            corner_radius=18,
            border_width=0,  # Remove double-border effect
        )
        section.pack(fill="x", expand=False, pady=(0, 18))
        title_label = ctk.CTkLabel(
            section,
            text=title,
            font=self.typography["h3"],
        )
        title_label.pack(anchor="w", padx=18, pady=(16, 8))
        return section

    def create_theme_and_sound_controls(self):
        """Add theme toggle and sound controls."""
        header_frame = ctk.CTkFrame(
            self.control_scroll_frame, fg_color="transparent"
        )
        header_frame.pack(fill="x", pady=(0, 18))

        title = ctk.CTkLabel(
            header_frame,
            text="Control Panel",
            font=self.typography["h2"],
        )
        title.pack(side="left")

        self.sound_enabled = True
        self.sound_button = ctk.CTkButton(
            header_frame,
            text="Sound: ON",
            width=80,
            height=28,
            fg_color=self.palette["accent_primary"][0],
            command=self.toggle_sound
        )
        self.sound_button.pack(side="right", padx=10)

        self.theme_switch = ctk.CTkSwitch(
            header_frame,
            text="Light Mode",
            command=self.toggle_theme,
            onvalue="light",
            offvalue="dark",
            font=self.typography["body_small"],
        )
        self.theme_switch.pack(side="right")

    def toggle_sound(self):
        """Toggle sound effects."""
        self.sound_enabled = not self.sound_enabled
        text = "Sound: ON" if self.sound_enabled else "Sound: OFF"
        self.sound_button.configure(text=text)

    def toggle_theme(self):
        """Switch between light and dark mode."""
        if self.theme_switch.get() == "light":
            ctk.set_appearance_mode("light")
        else:
            ctk.set_appearance_mode("dark")

    def create_data_section(self):
        """Construct the data loading section."""
        section = self._create_glass_section(self.control_scroll_frame, "Data Operations")

        self.load_btn = ctk.CTkButton(
            section,
            text="Load CSV Data",
            font=self.typography["body_bold"],
            height=40,
            fg_color=self.palette["accent_secondary"][0],
            hover_color=self.palette["accent_secondary"][1],
            command=self.load_data_handler,
        )
        self.load_btn.pack(fill="x", padx=18, pady=(0, 12))

        self.data_status_label = ctk.CTkLabel(
            section,
            text="No data loaded",
            font=self.typography["body_small"],
            text_color=self.palette["text_secondary"][0],
        )
        self.data_status_label.pack(anchor="w", padx=18, pady=(0, 12))
        
        # Data Preview (Mini Table)
        self.data_preview_frame = ctk.CTkFrame(section, fg_color="transparent", height=100)
        self.data_preview_frame.pack(fill="x", padx=18, pady=(0, 18))
        self.data_preview_label = ctk.CTkLabel(
            self.data_preview_frame,
            text="",
            font=self.typography["code"],
            justify="left",
            anchor="nw"
        )
        self.data_preview_label.pack(fill="both", expand=True)

    def load_data_handler(self):
        """Handle data loading."""
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        try:
            if file_path:
                self.engine.load_data(file_path)
            else:
                self.engine.load_data(None) # Generate mock data
            
        except DataLoadError as e:
            messagebox.showerror("Error", str(e))
        except Exception as e:
            logging.error(f"Unexpected error loading data: {e}")
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            
    def display_data_preview(self):
        """Show a snippet of the loaded data."""
        if self.engine.data is not None:
            preview_text = self.engine.data.head(5).to_string()
            self.data_preview_label.configure(text=preview_text)

    def create_model_section(self):
        """Construct the model training section."""
        section = self._create_glass_section(self.control_scroll_frame, "Model Training")

        model_row = ctk.CTkFrame(section, fg_color="transparent")
        model_row.pack(fill="x", padx=18, pady=(0, 12))

        ctk.CTkLabel(model_row, text="Select Model:").pack(side="left", padx=(0, 12))

        self.model_var = ctk.StringVar(value="Logistic Regression")
        self.model_menu = ctk.CTkOptionMenu(
            model_row,
            variable=self.model_var,
            values=[
                "Logistic Regression", 
                "Decision Tree Classifier",
                "Random Forest Classifier",
                "Gradient Boosting Classifier"
            ],
            width=220,
            command=lambda _: None,
        )
        self.model_menu.pack(side="left", padx=(0, 12))

        test_size_row = ctk.CTkFrame(section, fg_color="transparent")
        test_size_row.pack(fill="x", padx=18, pady=(0, 12))
        ctk.CTkLabel(test_size_row, text="Test Size:").pack(side="left", padx=(0, 12))
        self.test_size_var = ctk.StringVar(value="0.2")
        self.test_size_menu = ctk.CTkOptionMenu(
            test_size_row,
            variable=self.test_size_var,
            values=["0.1", "0.2", "0.3"],
            width=100,
        )
        self.test_size_menu.pack(side="left")

        self.train_btn = ctk.CTkButton(
            section,
            text="Train Model",
            font=self.typography["body_bold"],
            height=40,
            fg_color=self.palette["accent_primary"][0],
            hover_color=self.palette["accent_primary"][1],
            command=self.train_model,
        )
        self.train_btn.pack(fill="x", padx=18, pady=(0, 12))
        
        # Save/Load Buttons
        persistence_row = ctk.CTkFrame(section, fg_color="transparent")
        persistence_row.pack(fill="x", padx=18, pady=(0, 12))
        
        self.save_model_btn = ctk.CTkButton(
            persistence_row,
            text="Save Model",
            width=100,
            command=self.save_model_handler
        )
        self.save_model_btn.pack(side="left", padx=(0, 10), expand=True, fill="x")
        
        self.load_model_btn = ctk.CTkButton(
            persistence_row,
            text="Load Model",
            width=100,
            command=self.load_model_handler
        )
        self.load_model_btn.pack(side="left", expand=True, fill="x")

        self.training_status_label = ctk.CTkLabel(
            section,
            text="Model not trained",
            font=self.typography["body_small"],
            text_color=self.palette["text_secondary"][0],
        )
        self.training_status_label.pack(anchor="w", padx=18, pady=(0, 18))

    def train_model(self):
        """Train the selected model."""
        if self.engine.data is None:
            messagebox.showwarning("Warning", "Please load data first!")
            return
            
        # Update status
        self.training_status_label.configure(text="Training model...")
        self.train_btn.configure(state="disabled")
        
        # Capture values in main thread to ensure thread safety
        model_name = self.model_var.get()
        test_size = float(self.test_size_var.get())
        
        # Run training in a separate thread to keep GUI responsive
        threading.Thread(target=self._train_model_thread, args=(model_name, test_size), daemon=True).start()
        
    def _train_model_thread(self, model_name, test_size):
        """Train model in a separate thread."""
        try:
            self.engine.train_model(model_name, test_size)
            # The engine will notify observers when complete
            
        except Exception as e:
            logging.error(f"Training failed: {e}")
            self.after(0, lambda: messagebox.showerror("Error", f"Training failed: {str(e)}"))
            self.after(0, lambda: self.train_btn.configure(state="normal"))
            self.after(0, lambda: self.training_status_label.configure(text="Training failed"))

    def _training_complete(self):
        """Handle training completion."""
        self.training_status_label.configure(
            text=f"Training complete. Accuracy: {self.engine.accuracy:.1%}"
        )
        self.train_btn.configure(state="normal")
        
        # Update performance tab
        self.update_performance_tab()
        
        # Update threshold tuner plot
        self.update_threshold_tuner_plot()
        
        # Enable predictor and make initial prediction
        self.update_prediction()
        
    def save_model_handler(self):
        """Handle saving the model."""
        if self.engine.model is None:
            messagebox.showwarning("Warning", "No trained model to save!")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".joblib",
            filetypes=[("Joblib Files", "*.joblib")]
        )
        if file_path:
            try:
                self.engine.save_model(file_path)
                messagebox.showinfo("Success", f"Model saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save model: {e}")

    def load_model_handler(self):
        """Handle loading the model."""
        file_path = filedialog.askopenfilename(
            filetypes=[("Joblib Files", "*.joblib")]
        )
        if file_path:
            try:
                self.engine.load_model(file_path)
                # Note: Loading a model doesn't automatically load the training data used for it,
                # so some metrics/plots might not be available until new data is loaded/predicted.
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {e}")

    def create_prediction_section(self):
        """Construct the prediction parameters section."""
        section = self._create_glass_section(self.control_scroll_frame, "Prediction Parameters")

        # Route
        ctk.CTkLabel(section, text="Route:").pack(anchor="w", padx=18)
        self.route_var = ctk.StringVar(value="Campus Loop")
        ctk.CTkOptionMenu(
            section,
            variable=self.route_var,
            values=["Campus Loop", "Grove Village Route", "Al Hamra Link", "RAK City Express"],
        ).pack(fill="x", padx=18, pady=(0, 12))

        # Stop Distance
        ctk.CTkLabel(section, text="Stop Distance (km):").pack(anchor="w", padx=18)
        self.distance_slider = ctk.CTkSlider(section, from_=0, to=25, number_of_steps=25)
        self.distance_slider.set(5)
        self.distance_slider.pack(fill="x", padx=18, pady=(0, 12))

        # Time of Day
        ctk.CTkLabel(section, text="Time of Day (24h):").pack(anchor="w", padx=18)
        self.time_slider = ctk.CTkSlider(section, from_=0, to=23, number_of_steps=23)
        self.time_slider.set(12)
        self.time_slider.pack(fill="x", padx=18, pady=(0, 12))
        
        # Time Icon Canvas
        self.time_icon_canvas = tk.Canvas(section, width=30, height=30, bg=self.section_fg_color[1], highlightthickness=0)
        self.time_icon_canvas.pack(pady=(0, 12))
        self.time_slider.configure(command=lambda v: self.update_time_of_day_icon())
        self.update_time_of_day_icon()

        # Traffic
        ctk.CTkLabel(section, text="Traffic Condition:").pack(anchor="w", padx=18)
        self.traffic_var = ctk.StringVar(value="Medium")
        ctk.CTkOptionMenu(
            section,
            variable=self.traffic_var,
            values=["Low", "Medium", "High"],
        ).pack(fill="x", padx=18, pady=(0, 12))

        # Weather
        ctk.CTkLabel(section, text="Weather:").pack(anchor="w", padx=18)
        self.weather_var = ctk.StringVar(value="Clear")
        ctk.CTkOptionMenu(
            section,
            variable=self.weather_var,
            values=["Clear", "Cloudy", "Dusty", "Foggy", "Rainy"],
        ).pack(fill="x", padx=18, pady=(0, 12))

        # Special Event
        ctk.CTkLabel(section, text="Special Event:").pack(anchor="w", padx=18)
        self.event_var = ctk.StringVar(value="None")
        ctk.CTkOptionMenu(
            section,
            variable=self.event_var,
            values=["None", "Sports Match", "Career Fair", "University Exam", "Open Day"],
        ).pack(fill="x", padx=18, pady=(0, 18))

    def create_predict_button(self):
        """Add the main predict button."""
        self.predict_btn = ctk.CTkButton(
            self.control_scroll_frame,
            text="Predict Arrival",
            font=self.typography["h3"],
            height=50,
            fg_color=self.palette["accent_primary"][0],
            hover_color=self.palette["accent_primary"][1],
            command=self.update_prediction,
        )
        self.predict_btn.pack(fill="x", padx=24, pady=(0, 24))

    def create_simulation_deck(self):
        """Create the top-right simulation area."""
        self.sim_deck = ctk.CTkFrame(
            self.main_container,
            corner_radius=24,
            fg_color=self.panel_fg_color,
            border_width=1,
            border_color=self.panel_border_color,
        )
        self.sim_deck.grid(row=0, column=1, sticky="nsew", padx=0, pady=(0, 24))
        self.sim_deck.grid_rowconfigure(0, weight=0) # Header
        self.sim_deck.grid_rowconfigure(1, weight=1) # Canvas

        # Header
        header = ctk.CTkFrame(self.sim_deck, fg_color="transparent")
        header.pack(fill="x", padx=24, pady=18)
        
        ctk.CTkLabel(
            header,
            text="Live Simulation",
            font=self.typography["h2"],
        ).pack(side="left")

        self.sim_status_label = ctk.CTkLabel(
            header,
            text="Ready",
            font=self.typography["body"],
            text_color=self.palette["text_secondary"][0],
        )
        self.sim_status_label.pack(side="right")

        # Simulation Canvas
        self.sim_canvas = tk.Canvas(
            self.sim_deck,
            bg="#1e293b",
            highlightthickness=0,
        )
        self.sim_canvas.pack(fill="both", expand=True, padx=24, pady=(0, 24))
        
        # Draw initial path
        self.sim_canvas.bind("<Configure>", lambda e: self.draw_simulation_path())

    def draw_simulation_path(self):
        """Draw the curved path for the bus."""
        w = self.sim_canvas.winfo_width()
        h = self.sim_canvas.winfo_height()
        self.sim_canvas.delete("all")
        
        # Draw road
        points = []
        for i in range(self.config.SIMULATION_SEGMENTS + 1):
            t = i / self.config.SIMULATION_SEGMENTS
            x = w * 0.1 + (w * 0.8) * t
            y = h * 0.5 + (h * 0.3) * math.sin(t * math.pi * 2)
            points.append((x, y))
            
        self.sim_canvas.create_line(points, fill="#334155", width=40, capstyle="round", smooth=True)
        self.sim_canvas.create_line(points, fill="#475569", width=4, capstyle="round", smooth=True, dash=(10, 10))
        
        # Draw stops
        self.sim_canvas.create_oval(points[0][0]-8, points[0][1]-8, points[0][0]+8, points[0][1]+8, fill="#ef4444", outline="white", width=2)
        self.sim_canvas.create_oval(points[-1][0]-8, points[-1][1]-8, points[-1][0]+8, points[-1][1]+8, fill="#10b981", outline="white", width=2)

    def lerp(self, start, end, t):
        """Linear interpolation."""
        return start + (end - start) * t

    def _animate_bus_thread(self, speed, result):
        """Animation logic running in background thread."""
        w = self.sim_canvas.winfo_width()
        h = self.sim_canvas.winfo_height()
        
        steps = self.config.SIMULATION_SEGMENTS
        delay = 1.0 / (speed * 30) # Smoother animation
        
        points = []
        for i in range(steps + 1):
            t = i / steps
            x = w * 0.1 + (w * 0.8) * t
            y = h * 0.5 + (h * 0.3) * math.sin(t * math.pi * 2)
            points.append((x, y))

        for i in range(len(points) - 1):
            start_pos = points[i]
            end_pos = points[i+1]
            
            # Interpolate between points for smoothness
            sub_steps = 5
            for j in range(sub_steps):
                t = j / sub_steps
                x = self.lerp(start_pos[0], end_pos[0], t)
                y = self.lerp(start_pos[1], end_pos[1], t)
                self.after(0, lambda cx=x, cy=y: self.update_bus_position(cx, cy))
                time.sleep(delay / sub_steps)
            
        self.after(0, lambda: self.finish_simulation(result))

    def update_bus_position(self, x, y):
        """Update bus icon on canvas."""
        self.sim_canvas.delete("bus")
        
        # Draw a more detailed bus shape
        # Body
        self.sim_canvas.create_polygon(
            x-20, y-10, x+20, y-10, x+20, y+10, x-20, y+10,
            fill="#f59e0b", outline="white", width=1, tags="bus", smooth=True
        )
        # Windows
        self.sim_canvas.create_rectangle(x-15, y-8, x-5, y, fill="#bfdbfe", outline="", tags="bus")
        self.sim_canvas.create_rectangle(x+5, y-8, x+15, y, fill="#bfdbfe", outline="", tags="bus")
        # Wheels
        self.sim_canvas.create_oval(x-12, y+8, x-6, y+14, fill="#1e293b", outline="", tags="bus")
        self.sim_canvas.create_oval(x+6, y+8, x+12, y+14, fill="#1e293b", outline="", tags="bus")

    def finish_simulation(self, result):
        """Clean up after simulation."""
        self.is_simulating = False
        self.predict_btn.configure(state="normal")
        
        pred_text = result['prediction']
        conf = result['confidence']
        prob_on_time = result['on_time_prob']
        
        status_text = f"Prediction: {pred_text} ({conf:.1f}%) | P(On-Time): {prob_on_time:.1f}%"
        self.sim_status_label.configure(text=status_text)
        
        # Create environment effects
        self.create_environment_effects(result)
        
        # Play sound
        sound_file = Path(self.config.ASSETS_DIR) / "sounds" / ("success.wav" if pred_text == "On-Time" else "alert.wav")
        # Fallback to mp3 if wav not found (since we have mp3s)
        if not sound_file.exists():
             sound_file = Path(self.config.ASSETS_DIR) / "sounds" / ("on_time.mp3" if pred_text == "On-Time" else "delay_alert.mp3")
             
        if self.sound_enabled and sound_file.exists():
             play_sound_cross_platform(str(sound_file))

    def create_analytics_deck(self):
        """Create the bottom-right analytics area."""
        self.analytics_deck = ctk.CTkFrame(
            self.main_container,
            corner_radius=24,
            fg_color=self.panel_fg_color,
            border_width=1,
            border_color=self.panel_border_color,
        )
        self.analytics_deck.grid(row=1, column=1, sticky="nsew", padx=0, pady=0)
        
        # Header
        header = ctk.CTkFrame(self.analytics_deck, fg_color="transparent")
        header.pack(fill="x", padx=24, pady=18)
        
        ctk.CTkLabel(
            header,
            text="Analytics Dashboard",
            font=self.typography["h2"],
        ).pack(side="left")

        # Tabs
        self.tab_view = ctk.CTkTabview(
            self.analytics_deck,
            fg_color="transparent",
            segmented_button_fg_color=self.palette["panel_border"][1],
            segmented_button_selected_color=self.palette["accent_primary"][0],
            corner_radius=16,
        )
        self.tab_view.pack(fill="both", expand=True, padx=18, pady=(0, 18))
        
        self.tab_view.add("Model Performance")
        self.tab_view.add("Prediction Log")
        self.tab_view.add("Threshold Tuner")
        
        self.perf_content_frame = ctk.CTkScrollableFrame(self.tab_view.tab("Model Performance"), fg_color="transparent")
        self.perf_content_frame.pack(fill="both", expand=True)
        
        self.log_content_frame = ctk.CTkScrollableFrame(self.tab_view.tab("Prediction Log"), fg_color="transparent")
        self.log_content_frame.pack(fill="both", expand=True)
        
        self.tuner_content_frame = ctk.CTkFrame(self.tab_view.tab("Threshold Tuner"), fg_color="transparent")
        self.tuner_content_frame.pack(fill="both", expand=True)
        
        self.setup_threshold_tuner()

    def update_prediction(self):
        """Gather inputs and trigger prediction."""
        if self.is_simulating:
            return

        # Gather inputs
        input_record = {
            'route': self.route_var.get(),
            'stop_distance_km': self.distance_slider.get(),
            'day_of_week': "Monday",  # Simplified for demo
            'time_of_day': self.time_slider.get(),
            'traffic_condition': self.traffic_var.get(),
            'weather': self.weather_var.get(),
            'special_event': self.event_var.get(),
            'temperature_celsius': 30.0,  # Simplified
        }
        
        try:
            result = self.engine.predict(input_record)
            self.run_simulation(result)
            self.log_prediction(result)
        except ValueError as e:
            if "Model not trained" in str(e):
                messagebox.showwarning("Warning", "Please train or load a model first!")
            else:
                messagebox.showerror("Error", str(e))
        except Exception as e:
             messagebox.showerror("Error", f"Prediction failed: {e}")

    def run_simulation(self, prediction_result):
        """Run the visual bus simulation."""
        self.is_simulating = True
        self.predict_btn.configure(state="disabled")
        self.sim_status_label.configure(text="Simulating route...")
        
        # Determine speed based on prediction
        is_late = prediction_result['prediction'] == 'Late'
        speed = self.config.ANIMATION_SPEED_LATE if is_late else self.config.ANIMATION_SPEED_ON_TIME
        
        # Start animation thread
        threading.Thread(target=self._animate_bus_thread, args=(speed, prediction_result), daemon=True).start()

    def _animate_bus_thread(self, speed, result):
        """Animation logic running in background thread."""
        w = self.sim_canvas.winfo_width()
        h = self.sim_canvas.winfo_height()
        
        steps = self.config.SIMULATION_SEGMENTS
        delay = 1.0 / (speed * 10)
        
        for i in range(steps + 1):
            t = i / steps
            x = w * 0.1 + (w * 0.8) * t
            y = h * 0.5 + (h * 0.3) * math.sin(t * math.pi * 2)
            
            self.after(0, lambda cx=x, cy=y: self.update_bus_position(cx, cy))
            time.sleep(delay)
            
        self.after(0, lambda: self.finish_simulation(result))

    def update_bus_position(self, x, y):
        """Update bus icon on canvas."""
        self.sim_canvas.delete("bus")
        self.sim_canvas.create_rectangle(x-15, y-10, x+15, y+10, fill="#f59e0b", outline="white", tags="bus")
        self.sim_canvas.create_text(x, y, text="BUS", fill="black", font=("Arial", 8, "bold"), tags="bus")

    def finish_simulation(self, result):
        """Clean up after simulation."""
        self.is_simulating = False
        self.predict_btn.configure(state="normal")
        
        pred_text = result['prediction']
        conf = result['confidence']
        prob_on_time = result['on_time_prob']
        
        status_text = f"Prediction: {pred_text} ({conf:.1f}%) | P(On-Time): {prob_on_time:.1f}%"
        self.sim_status_label.configure(text=status_text)
        
        # Create environment effects
        self.create_environment_effects(result)
        
        # Play sound
        sound_file = Path("assets/sounds/success.wav") if pred_text == "On-Time" else Path("assets/sounds/alert.wav")
        if self.sound_enabled and sound_file.exists():
             play_sound_cross_platform(str(sound_file))

    def create_environment_effects(self, result):
        """Add weather/traffic effects to canvas."""
        self.sim_canvas.delete("effect")
        weather = result['context']['weather']
        
        if weather == "Rainy":
            for _ in range(50):
                x = random.randint(0, self.sim_canvas.winfo_width())
                y = random.randint(0, self.sim_canvas.winfo_height())
                self.sim_canvas.create_line(x, y, x+2, y+10, fill="#3b82f6", tags="effect")
        elif weather == "Foggy":
            self.sim_canvas.create_rectangle(0, 0, self.sim_canvas.winfo_width(), self.sim_canvas.winfo_height(), fill="#94a3b8", stipple="gray50", tags="effect")

    def log_prediction(self, result):
        """Add entry to prediction log."""
        entry_frame = ctk.CTkFrame(self.log_content_frame, fg_color=self.section_fg_color)
        entry_frame.pack(fill="x", pady=(0, 8))
        
        timestamp = time.strftime("%H:%M:%S")
        pred = result['prediction']
        color = self.palette["status_success"][0] if pred == "On-Time" else self.palette["status_danger"][0]
        
        ctk.CTkLabel(entry_frame, text=timestamp, font=self.typography["code"], width=80).pack(side="left", padx=8)
        ctk.CTkLabel(entry_frame, text=pred, font=self.typography["body_bold"], text_color=color).pack(side="left", padx=8)
        ctk.CTkLabel(entry_frame, text=f"Conf: {result['confidence']:.1f}%", font=self.typography["body_small"]).pack(side="right", padx=8)

        self.animate_log_entry(entry_frame)
        self.prediction_log_entries.append(entry_frame)

    def animate_log_entry(self, widget, step=0):
        """Slide log entries into place."""
        if step == 0:
            info = widget.pack_info()
            padx = info.get("padx", 0)
            pady = info.get("pady", 0)
            # Handle Tcl/Tk return values which can be strings or tuples
            if isinstance(padx, str): padx = int(float(padx))
            if isinstance(pady, str): pady = int(float(pady))
            if isinstance(padx, (tuple, list)): padx = int(padx[0])
            if isinstance(pady, (tuple, list)): pady = int(pady[0])
            
            widget._orig_padx = padx
            widget._orig_pady = pady
            widget.pack_configure(padx=(widget._orig_padx, widget._orig_padx), pady=(0, widget._orig_pady))

        if step >= 6:
            widget.pack_configure(padx=widget._orig_padx, pady=(widget._orig_pady, widget._orig_pady))
            return

        widget.pack_configure(pady=(step * 3, widget._orig_pady))
        self.after(30, lambda: self.animate_log_entry(widget, step + 1))

    def update_time_of_day_icon(self):
        """Draw a sun or moon icon based on the time slider."""
        if not hasattr(self, "time_icon_canvas"):
            return
        canvas = self.time_icon_canvas
        canvas.delete("all")
        hour = self.time_slider.get()
        
        if 6 <= hour <= 18:
            # Sun
            canvas.create_oval(5, 5, 25, 25, fill="#f59e0b", outline="")
        else:
            # Moon
            canvas.create_oval(5, 5, 25, 25, fill="#cbd5e1", outline="")
            canvas.create_oval(10, 5, 30, 25, fill=self.section_fg_color[1], outline="")

    def update_performance_tab(self):
        """Update the performance tab with animated results."""
        for widget in self.perf_content_frame.winfo_children():
            widget.destroy()

        if not hasattr(self.engine, 'y_test') or not hasattr(self.engine, 'y_pred') or self.engine.y_test is None:
            debug_label = ctk.CTkLabel(
                self.perf_content_frame,
                text="Train a model to see performance metrics.",
                font=self.typography["body_small"],
                text_color=self.palette["text_secondary"][0]
            )
            debug_label.pack(pady=10)
            return

        # Summary section with Metric Cards
        summary_frame = ctk.CTkFrame(self.perf_content_frame, fg_color="transparent")
        summary_frame.pack(fill="x", padx=10, pady=(0, 18))
        
        # Grid layout for cards
        summary_frame.grid_columnconfigure(0, weight=1)
        summary_frame.grid_columnconfigure(1, weight=1)
        summary_frame.grid_columnconfigure(2, weight=1)
        summary_frame.grid_columnconfigure(3, weight=1)

        accuracy = self.engine.accuracy * 100 if self.engine.accuracy else 0.0
        
        # Calculate other metrics if available
        precision = recall = f1 = 0.0
        if self.engine.y_test is not None and self.engine.y_pred is not None:
             report = classification_report(self.engine.y_test, self.engine.y_pred, output_dict=True)
             # Assuming 'Late' is the positive class we care about, or weighted avg
             if 'Late' in report:
                 precision = report['Late']['precision'] * 100
                 recall = report['Late']['recall'] * 100
                 f1 = report['Late']['f1-score'] * 100
             else:
                 precision = report['weighted avg']['precision'] * 100
                 recall = report['weighted avg']['recall'] * 100
                 f1 = report['weighted avg']['f1-score'] * 100

        MetricCard(summary_frame, "Accuracy", f"{accuracy:.1f}%", "up").grid(row=0, column=0, padx=5, sticky="ew")
        MetricCard(summary_frame, "Precision", f"{precision:.1f}%", "neutral").grid(row=0, column=1, padx=5, sticky="ew")
        MetricCard(summary_frame, "Recall", f"{recall:.1f}%", "neutral").grid(row=0, column=2, padx=5, sticky="ew")
        MetricCard(summary_frame, "F1 Score", f"{f1:.1f}%", "neutral").grid(row=0, column=3, padx=5, sticky="ew")
        
        # Buttons for plots
        btn_frame = ctk.CTkFrame(self.perf_content_frame, fg_color="transparent")
        btn_frame.pack(fill="x", pady=10)
        
        ctk.CTkButton(btn_frame, text="Confusion Matrix", command=self.show_confusion_matrix).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="ROC Curve", command=self.show_roc_curve).pack(side="left", padx=5)
        
        # Feature Importance Plot
        self.plot_feature_importance(self.perf_content_frame)

    def _apply_dark_theme(self, fig, ax):
        """Apply dark theme to matplotlib figure and axis."""
        bg_color = '#1e293b'
        text_color = 'white'
        
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        
        ax.spines['bottom'].set_color(text_color)
        ax.spines['top'].set_color(bg_color) 
        ax.spines['right'].set_color(bg_color)
        ax.spines['left'].set_color(text_color)
        
        ax.tick_params(axis='x', colors=text_color)
        ax.tick_params(axis='y', colors=text_color)
        
        ax.yaxis.label.set_color(text_color)
        ax.xaxis.label.set_color(text_color)
        ax.title.set_color(text_color)

    def plot_feature_importance(self, parent):
        """Plot feature importance bar chart."""
        importances = self.engine.get_feature_importances()
        if not importances:
            return

        features, values = zip(*importances)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        self._apply_dark_theme(fig, ax)
        
        sns.barplot(x=list(values), y=list(features), ax=ax, palette="viridis")
        ax.set_title("Feature Importance")
        ax.set_xlabel("Importance")
        
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="x", pady=10)

    def show_confusion_matrix(self):
        """Display confusion matrix in a new window."""
        if self.engine.y_test is None: return
        
        top = ctk.CTkToplevel(self)
        top.title("Confusion Matrix")
        top.geometry("600x500")
        top.configure(fg_color="#1e293b")
        
        fig, ax = plt.subplots(figsize=(6, 5))
        self._apply_dark_theme(fig, ax)
        
        cm = confusion_matrix(self.engine.y_test, self.engine.y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        # Fix heatmap text color
        for text in ax.texts:
            text.set_color("white" if int(text.get_text()) > cm.max()/2 else "black")
            
        ax.set_title(f'Confusion Matrix - {self.engine.model_name}')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def show_roc_curve(self):
        """Display ROC curve in a new window."""
        if self.engine.y_test is None: return
        
        # Only works for binary classification
        if len(self.engine.model.classes_) != 2:
            messagebox.showinfo("Info", "ROC Curve is only available for binary classification.")
            return

        top = ctk.CTkToplevel(self)
        top.title("ROC Curve")
        top.geometry("600x500")
        top.configure(fg_color="#1e293b")
        
        fig, ax = plt.subplots(figsize=(6, 5))
        self._apply_dark_theme(fig, ax)
        
        # Binarize labels: Late=1, On-Time=0
        y_test_bin = (self.engine.y_test == 'Late').astype(int)
        
        # Get probabilities for 'Late' class
        late_idx = list(self.engine.model.classes_).index('Late')
        y_score = self.engine.model.predict_proba(self.engine.X_test)[:, late_idx]
        
        fpr, tpr, _ = roc_curve(y_test_bin, y_score)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color='#38bdf8', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='#94a3b8', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc="lower right", facecolor='#1e293b', labelcolor='white')
        
        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def setup_threshold_tuner(self):
        """Setup the threshold tuner tab."""
        ctk.CTkLabel(self.tuner_content_frame, text="Probability Threshold Tuner", font=self.typography["h3"]).pack(pady=10)
        
        self.threshold_slider = ctk.CTkSlider(self.tuner_content_frame, from_=0, to=1, number_of_steps=100)
        self.threshold_slider.set(0.5)
        self.threshold_slider.pack(fill="x", padx=20, pady=10)
        
        self.threshold_label = ctk.CTkLabel(self.tuner_content_frame, text="Threshold: 0.50")
        self.threshold_label.pack()
        
        self.threshold_slider.configure(command=lambda v: self.threshold_label.configure(text=f"Threshold: {v:.2f}"))

    def update_threshold_tuner_plot(self):
        """Update precision-recall vs threshold plot."""
        for widget in self.tuner_content_frame.winfo_children():
            if isinstance(widget, (tk.Canvas, ctk.CTkCanvas)): # Remove old plot
                widget.destroy()
                
        if self.engine.y_test is None: return
        
        # Only works for binary classification
        if len(self.engine.model.classes_) != 2:
            return

        # Get probabilities for 'Late' class
        late_idx = list(self.engine.model.classes_).index('Late')
        y_score = self.engine.model.predict_proba(self.engine.X_test)[:, late_idx]
        
        # Binarize labels: Late=1, On-Time=0
        y_test_bin = (self.engine.y_test == 'Late').astype(int)
        
        precisions, recalls, thresholds = precision_recall_curve(y_test_bin, y_score)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        self._apply_dark_theme(fig, ax)
        
        ax.plot(thresholds, precisions[:-1], color="#38bdf8", linestyle="--", label="Precision")
        ax.plot(thresholds, recalls[:-1], color="#10b981", linestyle="-", label="Recall")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.set_title("Precision-Recall vs Threshold")
        ax.legend(loc="best", facecolor='#1e293b', labelcolor='white')
        ax.grid(True, color="#334155")
        
        canvas = FigureCanvasTkAgg(fig, master=self.tuner_content_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, pady=10)
