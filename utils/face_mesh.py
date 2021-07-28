import cv2
import mediapipe as mp


class FaceMesh:
    def __init__(self, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5,
                 max_num_faces=1):
        self. mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            max_num_faces=max_num_faces)

        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    def process(self, image):
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image.flags.writeable = False
        self.results = self.face_mesh.process(self.image)

        self.image.flags.writeable = True
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)

    def get_multi_face_landmarks(self):
        return self.results.multi_face_landmarks

    def draw_face_landmarks(self):
        for face_landmarks in self.get_multi_face_landmarks():
            self.mp_drawing.draw_landmarks(
                image=self.image,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACE_CONNECTIONS,
                landmark_drawing_spec=self.drawing_spec,
                connection_drawing_spec=self.drawing_spec)

        return self.image

    def get_glasses_landmarks(self):
        multi_face_landmarks = self.get_multi_face_landmarks()
        if multi_face_landmarks:
            faces = []
            for face_landmarks in multi_face_landmarks:
                lms = face_landmarks.landmark

                left_lm = lms[71]
                left_d_lm = lms[35]
                right_lm = lms[301]
                right_d_lm = lms[265]
                center_lm = lms[6]

                faces.append([left_lm, left_d_lm, right_lm, right_d_lm, center_lm])
            return faces
        else:
            return None

