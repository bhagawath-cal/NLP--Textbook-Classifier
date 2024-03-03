#!/usr/bin/env python3
import wx
import pandas
import numpy as np
import pickle
from glob import glob
from NNModel import FFNN


class MyFrame(wx.Frame):    
    def __init__(self):
        super().__init__(parent=None, title='NLP Helper')
        panel = wx.Panel(self)  
        outer_sizer = wx.BoxSizer(wx.VERTICAL)
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)  
        top_bar = wx.BoxSizer(wx.HORIZONTAL) 

        self.models_master = glob("models/*.pkl")
        if len(self.models_master) == 0:
            self.current_model = "Error: no models available"
        if "models/NNModel.pkl" not in self.models_master:
            self.current_model = self.models_master[0]
        else: 
            self.current_model = "models/NNModel.pkl"


        #Top bar 
        model = wx.StaticText(panel, label="Choose your Model:", size=(-1, -1))
        self.selector = wx.Choice(panel, choices=self.reorder_models(), size=(200, -1))
        self.selector.SetStringSelection(self.current_model) 
        self.selector.Bind(wx.EVT_CHOICE, self.handle_choice)
        self.refresh_btn = wx.Button(panel, label='refresh')
        self.refresh_btn.Bind(wx.EVT_BUTTON, self.refresh_models)
        top_bar.Add(model, 0, wx.ALL, 10)
        top_bar.Add(self.selector, 0, wx.ALL | wx.CENTER, 5)
        top_bar.Add(self.refresh_btn, 0, wx.ALL, 5)
        outer_sizer.Add(top_bar)



        # Text boxes 
        self.output = wx.TextCtrl(panel, size = (-1, 700), style = wx.TE_READONLY | wx.TE_MULTILINE | wx.TE_RICH )    
        self.output.SetMargins(5)
        self.text_ctrl = wx.TextCtrl(panel, style = wx.TE_PROCESS_ENTER)
        self.text_ctrl.Bind(wx.EVT_TEXT_ENTER, self.send_on_press)
        outer_sizer.Add(self.output, 0, wx.ALL | wx.EXPAND, 15)
        outer_sizer.Add(self.text_ctrl, 0, wx.ALL | wx.EXPAND, 15)        

        # Buttons
        send_btn = wx.Button(panel, label='send')
        send_btn.Bind(wx.EVT_BUTTON, self.send_on_press)
        button_sizer.Add(send_btn, 0, wx.ALL | wx.CENTER, 5)
        clear_btn = wx.Button(panel, label='clear')
        clear_btn.Bind(wx.EVT_BUTTON, self.clear_on_press)
        button_sizer.Add(clear_btn, 0, wx.ALL, 5)
        outer_sizer.Add(button_sizer, 0, wx.ALL | wx.CENTER, 5)        
       
        self.model = pickle.load(open(self.current_model, 'rb'))

        panel.SetSizer(outer_sizer)
        self.SetMinSize((500,900))   
        self.SetSize((500,900))
        self.text_ctrl.SetFocus()
        self.clear_text()
        self.Show()
        
    def handle_choice(self, event):
        selection = self.selector.GetCurrentSelection()
        selection = self.selector.GetString(selection)
        self.current_model = selection
        self.model = pickle.load(open(self.current_model, 'rb'))
        self.selector.SetItems(self.reorder_models())
        self.selector.SetStringSelection(self.current_model) 
        self.clear_text()
        

    def send_on_press(self, event):
        value = self.text_ctrl.GetValue()
        if not value:
            self.output.AppendText("You didn't enter anything!\n")
        else:
            self.output.AppendText(f'You typed:\n{value}\n\n')
            modelout = self.model.predict([value])
            self.output.AppendText(f'Model Output:\n')
            self.output.AppendText(f"I think you're referring to page {modelout[0]}!\nIf that doesn't seem correct, please try rephrasing your input\n")

            
        self.text_ctrl.SetValue("")
    
    def clear_on_press(self, event):
        self.clear_text()
        
    
    def clear_text(self):
        self.output.SetValue(f"Current model: {self.current_model}\n\nEnter your query below!\n")

    def refresh_models(self, event):
        self.models_master = sorted(glob("models/*.pkl"))
        self.selector.SetItems(self.reorder_models())
        self.selector.SetStringSelection(self.current_model)

    def reorder_models(self):
        "puts current selection at top of list"
        reordered = [elem for elem in self.models_master]
        reordered.remove(self.current_model)
        reordered.insert(0, self.current_model)
        return reordered


if __name__ == '__main__':
    app = wx.App()
    frame = MyFrame()
    app.MainLoop()
