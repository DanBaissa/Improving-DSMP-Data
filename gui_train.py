import tkinter as tk
from tkinter import filedialog
from main import main


class TrainModelGUI:
    def __init__(self, window):
        self.window = window
        window.title("Train Model GUI")

        self.DSMP_dir_label = tk.Label(window, text="DMSP data directory")
        self.DSMP_dir_label.pack()
        self.DSMP_dir_button = tk.Button(window, text="Browse", command=self.browse_DSMP_dir)
        self.DSMP_dir_button.pack()

        self.BM_dir_label = tk.Label(window, text="BM data directory")
        self.BM_dir_label.pack()
        self.BM_dir_button = tk.Button(window, text="Browse", command=self.browse_BM_dir)
        self.BM_dir_button.pack()

        self.conv_size_label = tk.Label(window, text="Convolution size")
        self.conv_size_label.pack()
        self.conv_size = tk.Entry(window)
        self.conv_size.pack()

        self.epochs_label = tk.Label(window, text="Number of epochs")
        self.epochs_label.pack()
        self.epochs = tk.Entry(window)
        self.epochs.pack()

        self.output_location_label = tk.Label(window, text="Output location")
        self.output_location_label.pack()
        self.output_location_button = tk.Button(window, text="Browse", command=self.browse_output_location)
        self.output_location_button.pack()

        self.train_button = tk.Button(window, text="Train model", command=self.train_model)
        self.train_button.pack()

    def browse_DSMP_dir(self):
        self.DSMP_dir = filedialog.askdirectory()

    def browse_BM_dir(self):
        self.BM_dir = filedialog.askdirectory()

    def browse_output_location(self):
        self.output_location = filedialog.askdirectory()

    def train_model(self):
        DSMP_dir = self.DSMP_dir
        BM_dir = self.BM_dir
        epochs = int(self.epochs.get())
        conv_size = int(self.conv_size.get())
        output_location = self.output_location
        main(DSMP_dir, BM_dir, epochs, conv_size, output_location)


if __name__ == "__main__":
    root = tk.Tk()
    TrainModelGUI(root)
    root.mainloop()