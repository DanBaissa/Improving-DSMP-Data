from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
import predict

def browse_button():
    # Allow user to select a directory and store it in global var
    global folder_path
    filename = filedialog.askdirectory()
    folder_path.set(filename)
    print(filename)

def submit_button():
    DSMP_folder = folder_path_DSMP.get()
    output_folder = folder_path_output.get()
    model_path = folder_path_model.get()
    predict.predict(DSMP_folder, output_folder, model_path)
    messagebox.showinfo("Information","Prediction completed")

root = Tk()
folder_path_DSMP = StringVar()
folder_path_output = StringVar()
folder_path_model = StringVar()
Label(root,text="DSMP Folder").grid(row=0, column=0)
Button(root, text="Browse", command=lambda: folder_path_DSMP.set(filedialog.askdirectory())).grid(row=0, column=1)
Label(root,text="Output Folder").grid(row=1, column=0)
Button(root, text="Browse", command=lambda: folder_path_output.set(filedialog.askdirectory())).grid(row=1, column=1)
Label(root,text="Model Path").grid(row=2, column=0)
Button(root, text="Browse", command=lambda: folder_path_model.set(filedialog.askopenfilename())).grid(row=2, column=1)
Button(root, text="Start Prediction", command=submit_button).grid(row=3, column=0, columnspan=2)

root.mainloop()
