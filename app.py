import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import time  # Import the time module for sleep functionality

# Load the model
model = tf.keras.models.load_model("test.keras")

# Define image dimensions and class names
img_height, img_width = 48, 48  # Adjust these to match your model's input size
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']


def process_image(image):
    # Resize and preprocess the image
    img = cv2.resize(image, (img_width, img_height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return img_array

def predict(image):
    # Make predictions
    predictions = model.predict(image)
    score = tf.nn.softmax(predictions[0])
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    return predicted_class, confidence

def main():
    st.title("ASL Hand Sign Detection")

    # Button to start and stop the webcam
    start_button = st.button('Start Webcam')
    stop_button = st.button('Stop Webcam')

    if start_button:
        cap = cv2.VideoCapture(0)

        # Create placeholders for the webcam feed and prediction
        video_placeholder = st.empty()
        prediction_placeholder = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image from webcam.")
                break

            # Display the webcam feed
            video_placeholder.image(frame, channels="BGR")

            # Process the image and make a prediction
            processed_image = process_image(frame)
            predicted_class, confidence = predict(processed_image)

            # Display the prediction
            prediction_placeholder.write(f"Predicted Sign: {predicted_class} | Confidence: {confidence:.2f}%")

            # Wait for 5 seconds before the next prediction
            time.sleep(5)

            # Stop the webcam when the user clicks the stop button
            if stop_button:
                break

        # Release the webcam
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
