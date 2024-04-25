import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog as fd
import tkinter.simpledialog
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sn

df = None
df_saved = None

def open_data():
    global df;
    file_name = fd.askopenfile()
    df = pd.read_csv(file_name, sep=',')
    refresh()
def save_data():
    file_name = fd.asksaveasfile()
    df.to_csv(file_name, index = False, line_terminator='\n')    
def add_data():
    global df;
    row = tk.simpledialog.askstring("Adding row", "Enter line of data with \',\': ")
    row = row.split(",")
    target_len = len(list(df.columns))
    if target_len > len(row):
        row = row[:target_len] + [0] * (target_len - len(row))
    df.loc[len(df)] = row
    refresh()
def del_data():
    dialog = tk.Toplevel()
    dialog.title('Delete')
    width = 200
    height = 75
    x = root.winfo_screenwidth() // 2 - width // 2
    y = root.winfo_screenheight() // 2 - height // 2
    dialog.geometry('{}x{}+{}+{}'.format(width, height, x, y))
    dialog.text = tk.Entry(dialog, borderwidth = 0)
    dialog.text.place(x=50, y=0, width=100)
    dialog.text.focus()
    dialog.button = tk.Button(dialog, text='Ok', borderwidth = 1, command=lambda: del_row(int(dialog.text.get())))
    dialog.button.place(x=50, y=25, width=100)
def del_row(row):
    global df;
    df = df.drop(row)
    df["#"] = np.arange(len(df))
    refresh()

def sort_data():
    global df, df_saved
    column = tk.simpledialog.askstring("Sorting", "Enter column for sorting: ")
    if df_saved is None:
        df_saved = df
    try:
        df = df.sort_values(column)
        info_label.config(text="Sorting is done")
    except KeyError:
        info_label.config(text="Unkhown name of the column")
    refresh()

def select_data():
    global df, df_saved
    column = tk.simpledialog.askstring("Selection", "Enter column for selecting: ")
    threshold = tk.simpledialog.askstring("Selection", "Enter threshold: ")
    relation = tk.simpledialog.askstring("Selection", "Enter relation (<, >, =): ")
    try:
        if df[column].dtypes == "float64":
            threshold = float(threshold)
        elif(df[column].dtypes == "int64"):
            threshold = int(threshold)
        if relation == '>' or relation == '<' or relation == '=':
            if df_saved is None:
                df_saved = df
            info_label.config(text="Selection is done")
        if relation == '>':
            df = df.loc[df[column] > threshold]
        if relation == '<':
            df = df.loc[df[column] < threshold]
        if relation == '=':
            df = df.loc[df[column] == threshold]
        refresh()
    except KeyError:
        info_label.config(text="Error in select attribute")

def cancel():
    global df, df_saved
    if df_saved is not None:
        df = df_saved
        df_saved = None
        info_label.config(text="All is Ok")
        refresh()    
    
def refresh():
    try:
        data_tab.tbl.destroy()
    except AttributeError:
        pass
    data_tab.tbl = tk.Frame(data_tab)
    headings = list(df.columns)
    attributes_number = len(headings)
    columns = list(range(attributes_number))
    data_tab.tbl.tree = ttk.Treeview(data_tab.tbl, show="headings", columns=columns)
    for i in range(attributes_number):
        data_tab.tbl.tree.heading(i, text=headings[i])
        data_tab.tbl.tree.column(i, minwidth=0, width=100)
    data_tab.tbl.ysb = ttk.Scrollbar(data_tab.tbl, orient="vertical",command=data_tab.tbl.tree.yview)
    data_tab.tbl.xsb = ttk.Scrollbar(data_tab.tbl, orient="horizontal",command=data_tab.tbl.tree.xview)
    data_tab.tbl.tree.configure(yscroll=data_tab.tbl.ysb.set,xscroll=data_tab.tbl.xsb.set)
    for i in range(df.shape[0]):
        data_tab.tbl.tree.insert("", "end", values=list(df.iloc[i]))
    data_tab.tbl.tree.grid(row=0, column=0, sticky="ns")
    data_tab.tbl.ysb.grid(row=0, column=1, sticky="ns")
    data_tab.tbl.xsb.grid(row=1, column=0, sticky="ew")
    data_tab.tbl.rowconfigure(0, weight=1)
    data_tab.tbl.columnconfigure(0, weight=1)
    data_tab.tbl.pack(side='bottom', fill='both', expand=1) 
def draw_bar():
    try:
        graph_tab.figure.canvas.get_tk_widget().destroy()
    except AttributeError:
        pass
    global df
    column = tk.simpledialog.askstring("Bar plot", "Enter name of column for visualization: ")
    values = pd.unique(df['Age'])
    values.sort()
    num_rows = df.shape[0]
    male_mean = values * [0.0]
    female_mean = values * [0.0]
    for i in range(len(values)):
        indices = [j for j in range(num_rows) if (df['Age'][j] == values[i] and df['Gender'][j] == 'Male')]
        male_mean[i] = np.mean(df[column][indices]) if (len(indices) > 0) else 0
        indices = [j for j in range(num_rows) if (df['Age'][j] == values[i] and df['Gender'][j] == 'Female')]
        female_mean[i] = np.mean(df[column][indices]) if (len(indices) > 0) else 0
    df2 = pd.DataFrame({'Male' : male_mean, 'Female' : female_mean})
    
    graph_tab.figure = plt.Figure(figsize = (6, 5))
    graph_tab.figure.ax = graph_tab.figure.add_subplot(111)
    graph_tab.figure.canvas = FigureCanvasTkAgg(graph_tab.figure, graph_tab)
    graph_tab.figure.canvas.get_tk_widget().pack(side='top', fill='x')
    df2.plot(kind='bar', ax=graph_tab.figure.ax, legend = True)
    
    x = np.arange(len(values))
    graph_tab.figure.ax.set_xticks(x, values)
    graph_tab.figure.ax.legend()
def draw_line():
    try:
        graph_tab.figure.canvas.get_tk_widget().destroy()
    except AttributeError:
        pass
    global df
    x_column = tk.simpledialog.askstring("Line plot", "Enter name of column for x: ")
    y_column = tk.simpledialog.askstring("Line plot", "Enter name of column for y: ")
    graph_tab.figure = plt.Figure(figsize = (6, 5))
    graph_tab.figure.ax = graph_tab.figure.add_subplot(111)
    graph_tab.figure.canvas = FigureCanvasTkAgg(graph_tab.figure, graph_tab)
    graph_tab.figure.canvas.get_tk_widget().pack(side='top', fill='x')
    df2 = df[[x_column, y_column]].groupby(x_column).mean()
    df2.plot(kind='line', ax=graph_tab.figure.ax, legend = True)
    

def draw_scatter():
    try:
        graph_tab.figure.canvas.get_tk_widget().destroy()
    except AttributeError:
        pass
    global df
    x_column = tk.simpledialog.askstring("Scatter plot", "Enter name of column for x: ")
    y_column = tk.simpledialog.askstring("Scatter plot", "Enter name of column for y: ")
    c_column = tk.simpledialog.askstring("Scatter plot", "Enter name of column for color: ")
    graph_tab.figure = plt.Figure(figsize = (6, 5))
    graph_tab.figure.ax = graph_tab.figure.add_subplot(111)
    graph_tab.figure.ax.scatter(df[x_column] + np.random.uniform(-0.3, +0.3, df.shape[0]),
                                df[y_column] + np.random.uniform(-0.3, +0.3, df.shape[0]),
                                c=df[c_column], alpha = 0.3, cmap='turbo')
    graph_tab.figure.canvas = FigureCanvasTkAgg(graph_tab.figure, graph_tab)
    graph_tab.figure.canvas.get_tk_widget().pack(side='top', fill='x')
def draw_corr():
    try:
        graph_tab.figure.canvas.get_tk_widget().destroy()
    except AttributeError:
        pass
    global df
    graph_tab.figure = plt.Figure(figsize = (8, 8))
    graph_tab.figure.ax = graph_tab.figure.add_subplot(111)
    sn.heatmap(df.corr(), square=True, ax=graph_tab.figure.ax)
    graph_tab.figure.canvas = FigureCanvasTkAgg(graph_tab.figure, graph_tab)
    graph_tab.figure.canvas.get_tk_widget().pack(side='top', fill='x')
    plt.setp(graph_tab.figure.ax.xaxis.get_majorticklabels(), rotation = 45)
    plt.setp(graph_tab.figure.ax.yaxis.get_majorticklabels(), rotation = 45)

root = tk.Tk()
tab_control = ttk.Notebook(root)
info_label = tk.Label(root, fg="white", bg="black", text='All is ok')

data_tab = ttk.Frame(tab_control)
tab_control.add(data_tab, text='Data Processing')
data_frame = ttk.Frame(data_tab)
data_tab.bopen = tk.Button(data_frame, bg="#C4FFB7", text='Open', command=open_data)
data_tab.bsave = tk.Button(data_frame, bg="#D7D3F3", text='Save', command=save_data)
data_tab.badd = tk.Button(data_frame, bg="#D7D7D7", text='Add row', command=add_data)
data_tab.bdel = tk.Button(data_frame, bg="#F76767", text='Delete row', command=del_data)
data_tab.bsort = tk.Button(data_frame, bg="#D7D7D7", text='Sort', command=sort_data)
data_tab.bsel = tk.Button(data_frame, bg="#D7D7D7", text='Select', command=select_data)
data_tab.bcan = tk.Button(data_frame, bg="#F76767", text='Cancel', command=cancel)

data_frame.pack(side='top', fill='x')
data_tab.bopen.pack(side='left', fill='x')
data_tab.bsave.pack(side='left', fill='x')
data_tab.badd.pack(side='left', fill='x')
data_tab.bdel.pack(side='left', fill='x')
data_tab.bsort.pack(side='left', fill='x')
data_tab.bsel.pack(side='left', fill='x')
data_tab.bcan.pack(side='left', fill='x')

graph_tab = ttk.Frame(tab_control)
tab_control.add(graph_tab, text='Visualization')
graph_frame = ttk.Frame(graph_tab)
graph_tab.bbar = tk.Button(graph_frame, text='Bar', command=draw_bar)
graph_tab.bline = tk.Button(graph_frame, text='Line', command=draw_line)
graph_tab.bscat = tk.Button(graph_frame, text='Scatter', command=draw_scatter)
graph_tab.bcorr = tk.Button(graph_frame, text='Correlation', command=draw_corr)


graph_frame.pack(side='top', fill='x')
graph_tab.bbar.pack(side='left', fill='x')
graph_tab.bline.pack(side='left', fill='x')
graph_tab.bscat.pack(side='left', fill='x')
graph_tab.bcorr.pack(side='left', fill='x')


ai_tab = ttk.Frame(tab_control)
tab_control.add(ai_tab, text='Artificial Intelligence')

tab_control.pack(expand=1, fill='both')
info_label.pack(side='bottom', fill='x')

root.title('Sleep Dependence On Age')
width = 600
height = 400
x = root.winfo_screenwidth() // 2 - width // 2
y = root.winfo_screenheight() // 2 - height // 2
root.geometry('{}x{}+{}+{}'.format(width, height, x, y))
root.mainloop()
