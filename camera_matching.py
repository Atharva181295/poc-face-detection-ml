import cv2
import face_recognition
from PIL import Image, ImageDraw

def load_and_encode_image(image_path):
    image = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(image)

    if not face_encodings:
        print(f"No face found in {image_path}")
        return None

    return face_encodings[0]

def compare_faces(known_face_encoding, unknown_face_encoding):
    results = face_recognition.compare_faces([known_face_encoding], unknown_face_encoding)
    return results[0]

def capture_image(camera_index=0):
    capture = cv2.VideoCapture(camera_index)

    if not capture.isOpened():
        print("Error: Could not open camera.")
        return None

    ret, frame = capture.read()

    if not ret:
        print("Error: Could not capture frame.")
        return None

    capture.release()

    return frame

def main():
    
    print("Press 'q' to capture image from camera (take 1)")
    input("Press Enter when ready...")
    known_image = capture_image()

    if known_image is None:
        return

    known_image_path = "known_person.jpg"
    cv2.imwrite(known_image_path, known_image)
    known_face_encoding = load_and_encode_image(known_image_path)

    if known_face_encoding is None:
        return

    
    print("Press 'q' to capture image from camera (take 2)")
    input("Press Enter when ready...")
    unknown_image = capture_image()

    if unknown_image is None:
        return

    unknown_image_path = "unknown_person.jpg"
    cv2.imwrite(unknown_image_path, unknown_image)
    unknown_face_encoding = load_and_encode_image(unknown_image_path)

    if unknown_face_encoding is None:
        return

    
    is_match = compare_faces(known_face_encoding, unknown_face_encoding)

    if is_match:
        print("These are the same person!")
    else:
        print("These are different people.")

if __name__ == "__main__":
    main()
