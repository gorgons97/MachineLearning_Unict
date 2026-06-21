import cv2
import torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from PIL import Image
from torchvision import transforms

import python_file.network as Network
import python_file.dataclass as StreetSign
from python_file.dirPath import modelliDir, yoloResult, yoloWeights, Test


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_height = 32
input_width = 32

preprocess = transforms.Compose([
    transforms.Resize((input_height, input_width)),
    transforms.ToTensor(), 
])

cls_transform = transforms.Compose([StreetSign.Rescale(32),StreetSign.RandomCrop(32),StreetSign.ToTensor()])

# mappa id -> nome classe
cls_map = {
    0: "Indicazione",
    1: "Divieto",
    2: "Pericolo"
}

def load_classifier(weights_path: str):
    model = Network.MiniAlexNetV2()
    model.load_state_dict(torch.load(modelliDir / 'MiniAlexNetV2_lr0.003_m0.6-1200.pth'))
    model.to(device)
    model.eval()
    return model, cls_transform


def detect_and_classify(
    img_path: str,
    det_weights: str = yoloWeights,
    cls_weights: str = modelliDir / 'MiniAlexNetV2_lr0.003_m0.6-1200.pth',
    save_img: bool = False,
    out_path: str = yoloResult / "testYolo.png",
    det_conf_th: float = 0.3
):
    """
    img_name: name of the image to process
    img_path: path dell'immagine di input
    det_weights: path dei pesi YOLO addestrati
    cls_weights: path dei pesi MiniAlexNet
    out_path: dove salvare l'immagine annotata
    det_conf_th: soglia minima di confidenza per tenere una detection YOLO
    """

    # 1) carica modello YOLO
    det_model = YOLO(det_weights)  # modello di detection

    # 2) carica il classificatore MiniAlexNet + trasformazioni    
    cls_model, cls_transform = load_classifier(cls_weights)
    cls_model.to(device)
    cls_model.eval()

    # 3) leggi immagine
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Impossibile leggere l'immagine: {img_path}")

    # YOLO lavora bene in RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 4) detection con YOLO
    # results è una lista, prendiamo il primo (una sola immagine)
    results = det_model(img_rgb)[0]

    # 5) loop sulle box trovate
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf_det = float(box.conf[0])

        # scarta box poco affidabili
        if conf_det < det_conf_th:
            continue

        # crop in RGB
        crop_rgb = img_rgb[y1:y2, x1:x2]
        if crop_rgb.size == 0:
            continue

        # 5a) preprocessing per MiniAlexNet: costruisci il "sample" come nel Dataset
        sample = {
            'image': np.array(crop_rgb),          # H x W x C, in RGB
            'landmarks': np.zeros((1, 2))        # dummy, non ti serve ma i transform lo chiedono
        }

        sample = cls_transform(sample)           # ora è un dict
        x = sample['image']                      # Tensor [C,H,W]
        x = x.unsqueeze(0)                       # [1,C,H,W]
        x = x.to(device).float()                 # <-- forza float32

        with torch.no_grad():
            logits = cls_model(x)
            probs = torch.softmax(logits, dim=1)[0]
            cls_id = int(torch.argmax(probs).item())
            conf_cls = float(probs[cls_id].item())
        
        cls_name = cls_map.get(cls_id, "sconosciuto")

        # etichetta da disegnare (poi puoi mappare cls_id a nome segnale)
        label = f"{cls_name} det:{conf_det:.2f} cls:{conf_cls:.2f}"

        # 6) disegna box e testo sull'immagine originale (BGR)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        text_thickness = 1

        # colore testo e background del testo
        text_color = (255, 255, 255)      # bianco
        text_bg_color = (0, 0, 255)       # rosso pieno

        # misura del testo
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)

        # posizione testo
        text_x = x1
        text_y = max(text_h + 5, y1 - 10)

        # rettangolo pieno dietro al testo
        cv2.rectangle(
            img_bgr,
            (text_x, text_y - text_h - baseline - 4),
            (text_x + text_w + 4, text_y + 4),
            text_bg_color,
            -1
        )

        # testo sopra il riquadro
        cv2.putText(
            img_bgr,
            label,
            (text_x + 2, text_y),
            font,
            font_scale,
            text_color,
            text_thickness,
            cv2.LINE_AA
        )

    if (save_img):
        # 7) salva risultato
        cv2.imwrite(str(out_path), img_bgr)
        print(f"Risultato salvato in {out_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    return img_pil

def draw_bounding_boxes(
    img_path, 
    results,
    save_img: bool = False,
    out_path: str = yoloResult / "testDiv.png"):
    
    # Load the image using OpenCV
    img_bgr = cv2.imread(str(img_path))
    
    # Draw bounding boxes on the image
    for box in results:
        x1, y1, x2, y2, cls_name, conf_det = box

        # etichetta da disegnare (poi puoi mappare cls_name a nome segnale)
        label = f"{cls_name} det:{conf_det:.2f}"
        
        cv2.rectangle(img_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        text_thickness = 1

        # colore testo e background del testo
        text_color = (255, 255, 255)      # bianco
        text_bg_color = (0, 0, 255)       # rosso pieno

        # misura del testo
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)

        # posizione testo
        text_x = x1
        text_y = max(text_h + 5, y1 - 10)

        # rettangolo pieno dietro al testo
        cv2.rectangle(
            img_bgr,
            (text_x, text_y - text_h - baseline - 4),
            (text_x + text_w + 4, text_y + 4),
            text_bg_color,
            -1
        )

        # testo sopra il riquadro
        cv2.putText(
            img_bgr,
            label,
            (text_x + 2, text_y),
            font,
            font_scale,
            text_color,
            text_thickness,
            cv2.LINE_AA
        )
        
    if (save_img):
        #salva risultato
        cv2.imwrite(str(out_path), img_bgr)
        print(f"Risultato salvato in {out_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    return img_pil