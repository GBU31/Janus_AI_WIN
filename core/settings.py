import json
import tkinter as tk

class SettingsTkGUI:
    def __init__(self, settings_file_path = "core/settings.json") -> None:
        self.settings_file_path = settings_file_path


    def settings_json(self) -> dict:
        with open(self.settings_file_path, 'r') as file:
            data = json.load(file)

        return data
    

    def Def_Settings(self):
        with open(self.settings_file_path, 'w') as sf:
            Default = {"width": "640", "height": '480'}
            jd = json.dumps(Default)
            sf.write(jd)

        return self.settings_json()
    
    def get_settings(self) -> dict:
        try:
            settingsFile = self.settings_json()
        except json.decoder.JSONDecodeError:
            settingsFile = self.Def_Settings()
            
        
        with open(self.settings_file_path, 'r') as sf:
            data = json.load(sf)

        return data
    
    def run_settings(self):
        data = self.get_settings()

        root = tk.Tk()
        root.geometry("400x200")
        widthLabel = tk.Label(root, text="width:")
        widthLabel.pack()
        
        width = tk.Entry(root)
        width.insert(0, data["width"])
        width.pack()

        heightLabel = tk.Label(root, text="height:")
        heightLabel.pack()
        
        height = tk.Entry(root)
        height.insert(0, data['height'])
        height.pack()

        def on_click():
            data = {"width": width.get(), "height":height.get()}
            jd = json.dumps(data)

            with open(self.settings_file_path, 'w') as file:
                file.write(jd)
            

        button = tk.Button(root, text="Save", command=on_click)
        button.pack()

        root.mainloop()