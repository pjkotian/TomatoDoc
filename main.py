import cv2
import numpy as np
import sqlite3
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.lang import Builder
from kivy.clock import Clock
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.spinner import Spinner
from kivy.uix.boxlayout import BoxLayout
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import load_model
import pickle
from joblib import load

class CastLayer(Layer):
    def call(self, inputs):
        return tf.cast(inputs, dtype=tf.float32)

# Load the model with custom objects
xception_model = load_model("assets/models/tomato_disease_xception.h5", custom_objects={"Cast": CastLayer})
with open("assets/models/rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

fertilizer_data = load("assets/models/fertilizer_data.pkl")

# SQLite Database Setup
DB_NAME = "fertilizer_data.db"

def create_database():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT,
            disease TEXT,
            fertilizer TEXT,
            category TEXT,
            remarks TEXT,
            bis_reference TEXT
        )
    ''')
    conn.commit()
    conn.close()

class SplashScreen(Screen):
    def schedule_switch(self):
        Clock.schedule_once(self.switch_to_home, 10)

    def switch_to_home(self, dt):
        self.manager.current = "home"

class HomeScreen(Screen):
    pass

class CameraScreen(Screen):
    def capture_image(self):
        camera = self.ids.camera
        image_path = "captured.png"
        camera.export_to_png(image_path)
        App.get_running_app().predict_fertilizer(image_path)

class UploadScreen(Screen):
    def load_image(self, filepath):
        if filepath:
            App.get_running_app().predict_fertilizer(filepath[0])

class FertilizerInputScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.elements = []

    def set_disease(self, predicted_disease, image_path, fertilizer_name="Not Assigned", category="-", remarks="-", bis_ref="-"):
        """ Store the predicted disease and image path for later use. """
        self.predicted_disease = predicted_disease
        self.image_path = image_path
        self.fertilizer_name = fertilizer_name
        self.category = category
        self.remarks = remarks
        self.bis_ref = bis_ref

        # Ensure the label exists before updating it
        if 'disease_label' in self.ids:
            self.ids.disease_label.text = f"Disease: {predicted_disease}"
        else:
            print("Warning: 'disease_label' not found in UI!")

    def add_element(self):
        element_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
        spinner = Spinner(text='Select Element', values=['Sulfur','Calcium','Magnesium','Zinc','Boron','Iron','Copper','Manganese','Aluminium','Molybdenum'])
        input_field = TextInput(hint_text='Enter Value', multiline=False, size_hint_x=0.6)
        remove_button = Button(text='Remove', size_hint_x=0.3)
        remove_button.bind(on_release=lambda btn: self.remove_element(element_layout))
        
        element_layout.add_widget(spinner)
        element_layout.add_widget(input_field)
        element_layout.add_widget(remove_button)
        self.ids.elements_box.add_widget(element_layout)
        self.elements.append(element_layout)
    
    def remove_element(self, element_layout):
        self.ids.elements_box.remove_widget(element_layout)
        self.elements.remove(element_layout)

    def submit_fertilizer_data(self):
        npk_values = {
            "Nitrogen": self.ids.nitrogen_input.text,
            "Phosphorus": self.ids.phosphorus_input.text,
            "Potassium": self.ids.potassium_input.text,
        }
        additional_elements = []
        for layout in self.elements:
            element_name = layout.children[2].text  # Spinner
            element_value = layout.children[1].text  # TextInput
            additional_elements.append((element_name, element_value))

        # Fetch recommended fertilizer based on user input and disease
        matching_fertilizers = fertilizer_data[fertilizer_data[self.predicted_disease] == 1]

        if not matching_fertilizers.empty:
            best_fertilizer = matching_fertilizers.iloc[0]  # Select first match
            fertilizer_name = best_fertilizer["Fertilizer Name"]
            category = best_fertilizer["Category"]
            remarks = best_fertilizer["Remarks"]
            bis_ref = best_fertilizer["BIS Standard Reference"]
        else:
            fertilizer_name = "No matching fertilizer found"
            category = remarks = bis_ref = "-"

        # Get app instance and result screen
        app = App.get_running_app()
        result_screen = app.root.get_screen("result")

        # Save to SQLite before switching to result screen
        app.save_to_database(
            self.image_path, self.predicted_disease, fertilizer_name, category, remarks, bis_ref
        )

        # Ensure all values are passed correctly to result screen
        result_screen.update_results(
            self.image_path,
            self.predicted_disease,
            fertilizer_name,
            category,
            remarks,
            bis_ref
        )

        app.root.current = "result"  # Transition to Result Screen

class ResultScreen(Screen):
    def update_results(self, image_path, disease, fertilizer, category, remarks, bis_ref):
        if image_path:
            self.ids.result_image.source = image_path
        self.ids.disease_label.text = f"Disease: {disease}"
        self.ids.fertilizer_label.text = f"Fertilizer: {fertilizer}"
        self.ids.category_label.text = f"Category: {category}"
        self.ids.remarks_label.text = f"Remarks: {remarks}"
        self.ids.bis_label.text = f"BIS Reference: {bis_ref}"

# Load Kivy UI from .kv file
Builder.load_file("appdesign.kv")

class TomatoApp(App):
    def build(self):
        create_database()  # Ensure database is created
        self.screen_manager = ScreenManager()
        self.screen_manager.add_widget(SplashScreen(name="splash"))
        self.screen_manager.add_widget(HomeScreen(name="home"))
        self.screen_manager.add_widget(CameraScreen(name="camera"))
        self.screen_manager.add_widget(UploadScreen(name="upload"))
        self.screen_manager.add_widget(FertilizerInputScreen(name="fertilizer_input"))
        self.screen_manager.add_widget(ResultScreen(name="result"))
        return self.screen_manager

    def switch_to_home(self, dt):
        self.screen_manager.current = "home"

    def predict_fertilizer(self, image_path):
        image = cv2.imread(image_path)
        test_image = cv2.resize(image, (299, 299))
        test_image = np.expand_dims(test_image, axis=0) / 255.0

        # Predict Disease
        prediction = xception_model.predict(test_image)
        predicted_class = np.argmax(prediction)
        class_labels = {0: 'bacterial_spot', 1: 'early_blight', 2: 'healthy', 3: 'late_blight',
                        4: 'leaf_mold', 6: 'septoria_leaf_spot', 8: 'twospotted_spider_mite',
                        7: 'target_spot', 5: 'mosaic_virus', 9: 'yellow_leaf_curl_virus'}
        predicted_disease = class_labels.get(predicted_class, "Unknown")

        if predicted_disease.lower() != "healthy":
            # Pass only required parameters (Default values for others)
            fertilizer_input_screen = self.root.get_screen("fertilizer_input")
            fertilizer_input_screen.set_disease(predicted_disease, image_path)
            self.root.current = "fertilizer_input"  # Ensure transition to input screen

        else:
            # If healthy, save directly & switch to result screen
            self.save_to_database(image_path, "Healthy", "No Fertilizer Needed", "-", "-", "-")
            result_screen = self.root.get_screen("result")
            result_screen.update_results(image_path, "Healthy", "No Fertilizer Needed", "-", "-", "-")
            self.root.current = "result"  # Show result immediately if healthy

    def save_to_database(self, image_path, disease, fertilizer_name, category, remarks, bis_ref):
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO recommendations (image_path, disease, fertilizer_name, category, remarks, bis_reference)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (image_path, disease, fertilizer_name, category, remarks, bis_ref))
        conn.commit()
        conn.close()

if __name__ == "__main__":
    TomatoApp().run()