"""
Document Detector - детектирует подписи, печати и QR-коды
Адаптирован для работы с локальными YOLOv8 моделями без API
Оптимизирован для GTX 1650 Ti (4GB VRAM)
"""

import os
import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json
import torch

# Отключаем лимит пикселей для больших изображений
Image.MAX_IMAGE_PIXELS = None


class DocumentDetector:
    """Основной класс для детекции элементов на документах"""
    
    def __init__(self, models_dir: str = "models"):
        """
        Инициализация детектора
        
        Args:
            models_dir: путь к папке с моделями
        """
        self.models_dir = Path(models_dir)
        self.qr_reader = None
        self.stamp_model = None
        self.signature_model = None
        
        # Параметры обработки (оптимизировано для GTX 1650 Ti)
        self.dpi = 400 # Снижено для экономии VRAM (было 200)
        self.confidence_threshold = 0.5
        self.aspect_ratio_range = (0.7, 1.3)
        
        # Проверяем доступность GPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"✓ VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("⚠ GPU not available, using CPU")
        
        # Инициализируем детекторы
        self._init_detectors()
    
    def _init_detectors(self):
        """Инициализация всех детекторов"""
        
        # 1. QR детектор (легкий, без GPU)
        try:
            from qreader import QReader
            self.qr_reader = QReader(model_size='l')
            print("✓ QR detector loaded (QReader)")
        except Exception as e:
            print(f"⚠ QR detector not available: {e}")
        
        # 2. Stamp detector (YOLOv8)
        try:
            from ultralytics import YOLO
            stamp_path = self.models_dir / "stamp_model.pt"
            
            if stamp_path.exists():
                self.stamp_model = YOLO(str(stamp_path))
                self.stamp_model.to(self.device)
                print(f"✓ Stamp detector loaded (YOLOv8 on {self.device})")
            else:
                print(f"⚠ Stamp model not found at: {stamp_path}")
                print("  Expected: models/stamp_model.pt")
        except Exception as e:
            print(f"⚠ Stamp detector error: {e}")
        
        # 3. Signature detector (YOLOv8)
        try:
            from ultralytics import YOLO
            sign_path = self.models_dir / "signature_model.pt"
            
            if sign_path.exists():
                self.signature_model = YOLO(str(sign_path))
                self.signature_model.to(self.device)
                print(f"✓ Signature detector loaded (YOLOv8 on {self.device})")
            else:
                print(f"⚠ Signature model not found at: {sign_path}")
                print("  Expected: models/signature_model.pt")
        except Exception as e:
            print(f"⚠ Signature detector error: {e}")

    def process_file(self, file_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Обработка файла с приведением к целевому формату JSON
        """
        file_path = Path(file_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing: {file_path.name}")

        # 1. Загрузка изображений
        try:
            if file_path.suffix.lower() == '.pdf':
                images = self._pdf_to_images(file_path)
            else:
                img = cv2.imread(str(file_path))
                if img is None: raise ValueError("Could not read image")
                images = [img]
        except Exception as e:
            return {"filename": file_path.name, "status": "error", "error": str(e)}

        # Подготовка структуры для целевого формата
        # Формат: { "filename.pdf": { "page_1": { ... } } }
        final_json = {
            file_path.name: {}
        }

        processed_images = []
        global_annotation_id = 1  # Сквозной счетчик для annotation_X

        # 2. Обработка страниц
        for page_num, image in enumerate(images):
            # Получаем размеры страницы для JSON
            page_h, page_w = image.shape[:2]

            # Детекция
            detections = self._detect_on_image(image, page_num + 1)

            # Формируем структуру для текущей страницы
            page_key = f"page_{page_num + 1}"
            page_data = {
                "annotations": [],
                "page_size": {
                    "width": page_w,
                    "height": page_h
                }
            }

            # Преобразуем детекции в нужный формат
            for det in detections:
                # Преобразование типа
                cat = det["type"]
                if cat == "qr_code": cat = "qr"  # В задании требуется "qr", а не "qr_code"

                # Координаты
                x1, y1, x2, y2 = det["bbox"]
                w = x2 - x1
                h = y2 - y1

                annotation_obj = {
                    f"annotation_{global_annotation_id}": {
                        "category": cat,
                        "bbox": {
                            "x": x1,
                            "y": y1,
                            "width": w,
                            "height": h
                        },
                        "area": float(w * h)
                    }
                }
                page_data["annotations"].append(annotation_obj)
                global_annotation_id += 1

            final_json[file_path.name][page_key] = page_data

            # Отрисовка для визуализации (оставляем для удобства, но в JSON не пишем)
            annotated_image = self._draw_detections(image.copy(), detections)
            processed_images.append(annotated_image)

            images[page_num] = None

            # Очистка памяти
        del images
        import gc
        gc.collect()

        # 3. Сохранение визуализации (картинка/pdf с рамками)
        output_filename = file_path.stem + "_processed"
        output_path = None

        if file_path.suffix.lower() == '.pdf':
            output_path = output_dir / f"{output_filename}.pdf"
            self._save_as_pdf(processed_images, output_path)
        else:
            output_path = output_dir / f"{output_filename}.jpg"
            cv2.imwrite(str(output_path), processed_images[0])

        del processed_images
        gc.collect()

        # 4. Сохранение JSON в ТРЕБУЕМОМ формате
        json_path = output_dir / f"{output_filename}.json"

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(final_json, f, ensure_ascii=False, indent=2)

        # Возвращаем данные для фронтенда (можно вернуть чуть упрощенную структуру для UI,
        # но JSON на диске будет правильный)
        return {
            "filename": file_path.name,
            "output_image": output_path.name,
            "formatted_json": final_json,  # Отдаем новый формат
            "results": final_json,  # Для совместимости
            "status": "success"
        }
    
    def _pdf_to_images(self, pdf_path: Path) -> List[np.ndarray]:
        """Конвертирует PDF в список изображений"""
        try:
            poppler_path = r"C:\poppler\poppler-25.11.0\Library\bin"
            pages = convert_from_path(str(pdf_path), dpi=self.dpi, poppler_path=poppler_path)
            return [cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR) for page in pages]
        except Exception as e:
            print(f"Error converting PDF: {e}")
            return []
    
    def _detect_on_image(self, image: np.ndarray, page_num: int) -> List[Dict]:
        """Детекция всех элементов на одном изображении"""
        detections = []
        
        # 1. Детекция QR-кодов
        qr_detections = self._detect_qr_codes(image, page_num)
        detections.extend(qr_detections)
        
        # 2. Детекция печатей
        stamp_detections = self._detect_stamps(image, page_num)
        detections.extend(stamp_detections)
        
        # 3. Детекция подписей (когда модель будет готова)
        signature_detections = self._detect_signatures(image, page_num)
        detections.extend(signature_detections)
        
        return detections
    
    def _detect_qr_codes(self, image: np.ndarray, page_num: int) -> List[Dict]:
        """Детекция QR-кодов"""
        if not self.qr_reader:
            return []
        
        detections = []
        
        try:
            qr_detections = self.qr_reader.detect(image=image)
            
            if not qr_detections:
                return []
            
            for detection in qr_detections:
                if detection is None:
                    continue
                
                # Разбор detection
                if isinstance(detection, dict):
                    score = detection.get('confidence', 1.0)
                    bbox = detection.get('bbox_xyxy', detection.get('bbox'))
                else:
                    score = 1.0
                    bbox = detection
                
                if bbox is None:
                    continue
                
                # Фильтр по уверенности
                if score < self.confidence_threshold:
                    continue
                
                x1, y1, x2, y2 = map(int, bbox)
                w = abs(x2 - x1)
                h = abs(y2 - y1)
                
                if h == 0:
                    continue
                
                # Фильтр по геометрии
                ratio = w / h
                if not (self.aspect_ratio_range[0] < ratio < self.aspect_ratio_range[1]):
                    continue
                
                # Фильтр по цвету (отсекаем цветные печати)
                crop = image[y1:y2, x1:x2]
                if self._is_colored_stamp(crop):
                    continue
                
                detections.append({
                    "type": "qr_code",
                    "page": page_num,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(score),
                    "width": w,
                    "height": h
                })
        
        except Exception as e:
            print(f"QR detection error: {e}")
        
        return detections
    
    def _detect_stamps(self, image: np.ndarray, page_num: int) -> List[Dict]:
        """Детекция печатей/штампов с YOLOv8"""
        if not self.stamp_model:
            return []
        
        detections = []
        
        try:
            # YOLOv8 inference
            results = self.stamp_model(
                image,
                conf=self.confidence_threshold,
                verbose=False
            )[0]
            
            # Обрабатываем результаты
            if results.boxes is not None:
                for box in results.boxes:
                    # Координаты (xyxy format)
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    
                    # Уверенность
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    # Класс
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = results.names[class_id] if hasattr(results, 'names') else 'stamp'
                    
                    # Размеры
                    w = x2 - x1
                    h = y2 - y1
                    
                    detections.append({
                        "type": "stamp",
                        "page": page_num,
                        "bbox": [x1, y1, x2, y2],
                        "confidence": confidence,
                        "width": w,
                        "height": h,
                        "class": class_name
                    })
        
        except Exception as e:
            print(f"Stamp detection error: {e}")
        
        return detections
    
    def _detect_signatures(self, image: np.ndarray, page_num: int) -> List[Dict]:
        """Детекция подписей с YOLOv8"""
        if not self.signature_model:
            return []
        
        detections = []
        
        try:
            # YOLOv8 inference
            results = self.signature_model(
                image,
                conf=self.confidence_threshold,
                verbose=False
            )[0]
            
            # Обрабатываем результаты
            if results.boxes is not None:
                for box in results.boxes:
                    # Координаты (xyxy format)
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    
                    # Уверенность
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    # Класс
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = results.names[class_id] if hasattr(results, 'names') else 'signature'
                    
                    # Размеры
                    w = x2 - x1
                    h = y2 - y1
                    
                    detections.append({
                        "type": "signature",
                        "page": page_num,
                        "bbox": [x1, y1, x2, y2],
                        "confidence": confidence,
                        "width": w,
                        "height": h,
                        "class": class_name
                    })
        
        except Exception as e:
            print(f"Signature detection error: {e}")
        
        return detections
    
    def _is_colored_stamp(self, image_crop: np.ndarray, saturation_threshold: int = 25) -> bool:
        """
        Проверка на цветную печать
        Returns True если фрагмент цветной (печать)
        Returns False если ч/б (QR код)
        """
        if image_crop is None or image_crop.size == 0:
            return False
        
        try:
            hsv = cv2.cvtColor(image_crop, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1]
            mean_sat = np.mean(saturation)
            return mean_sat > saturation_threshold
        except:
            return False
    
    def _draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Рисует bounding boxes на изображении"""
        # Цвета для разных типов
        colors = {
            "qr_code": (0, 255, 0),      # Зеленый
            "stamp": (255, 0, 0),         # Синий (BGR)
            "signature": (0, 0, 255)      # Красный
        }
        
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            det_type = det["type"]
            confidence = det["confidence"]
            
            color = colors.get(det_type, (255, 255, 255))
            
            # Рисуем прямоугольник
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            
            # Добавляем метку
            label = f"{det_type}: {confidence:.2f}"
            cv2.putText(
                image, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )
        
        return image
    
    def _save_as_pdf(self, images: List[np.ndarray], output_path: Path):
        """Сохранение списка изображений как PDF"""
        pil_images = [
            Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            for img in images
        ]
        
        if pil_images:
            pil_images[0].save(
                str(output_path),
                "PDF",
                resolution=100.0,
                save_all=True,
                append_images=pil_images[1:]
            )


# Тестирование
if __name__ == "__main__":
    detector = DocumentDetector()
    print("Detector initialized successfully!")
