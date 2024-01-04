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

def main():
    
    known_image_path = "/home/atharva/Workspace/Machine Learning/face_recognition/big-b-new.jpg"
    known_face_encoding = load_and_encode_image(known_image_path)

    if known_face_encoding is None:
        return

    
    unknown_image_path = "/home/atharva/Workspace/Machine Learning/face_recognition/goggles.jpg"
    unknown_face_encoding = load_and_encode_image(unknown_image_path)

    if unknown_face_encoding is None:
        return

    is_match = compare_faces(known_face_encoding, unknown_face_encoding)

    image = Image.open(unknown_image_path)
    draw = ImageDraw.Draw(image)

    if is_match:
        print("These are the same person!")
        draw.text((10, 10), "Match!", fill="green")
    else:
        print("These are different people.")
        draw.text((10, 10), "No Match!", fill="red")

    # image.show()

if __name__ == "__main__":
    main()
