#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import onnxruntime as rt
import cv2  # Used for preprocessing and postprocessing using OpenCV


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class ObjectDetectorONNX:
    def __init__(self, config):
        self.architecture = config["architecture"]
        self.model = config["model"]
        self.model_dir_path = config["model_dir_path"]
        self.checkpoint = config["checkpoint"]
        self.device = config["device"]
        self.confident = config["confident"]
        self.iou = config["iou"]
        self.classes = config["classes"]
        self.multiple_instance = config["multiple_instance"]
        # "" / "auto" -> infer from the ONNX metadata; "detect"/"segment" force it.
        self.task_override = (config.get("task") or "").lower()
        self.detector = None
        # Set once the model is loaded: True for segmentation models, in which
        # case detect() also returns one boolean instance mask per detection.
        self.is_segment = False
        self.class_dict = {
            0: "person",
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            4: "airplane",
            5: "bus",
            6: "train",
            7: "truck",
            8: "boat",
            9: "traffic light",
            10: "fire hydrant",
            11: "stop sign",
            12: "parking meter",
            13: "bench",
            14: "bird",
            15: "cat",
            16: "dog",
            17: "horse",
            18: "sheep",
            19: "cow",
            20: "elephant",
            21: "bear",
            22: "zebra",
            23: "giraffe",
            24: "backpack",
            25: "umbrella",
            26: "handbag",
            27: "tie",
            28: "suitcase",
            29: "frisbee",
            30: "skis",
            31: "snowboard",
            32: "sports ball",
            33: "kite",
            34: "baseball bat",
            35: "baseball glove",
            36: "skateboard",
            37: "surfboard",
            38: "tennis racket",
            39: "bottle",
            40: "wine glass",
            41: "cup",
            42: "fork",
            43: "knife",
            44: "spoon",
            45: "bowl",
            46: "banana",
            47: "apple",
            48: "sandwich",
            49: "orange",
            50: "broccoli",
            51: "carrot",
            52: "hot dog",
            53: "pizza",
            54: "donut",
            55: "cake",
            56: "chair",
            57: "couch",
            58: "potted plant",
            59: "bed",
            60: "dining table",
            61: "toilet",
            62: "tv",
            63: "laptop",
            64: "mouse",
            65: "remote",
            66: "keyboard",
            67: "cell phone",
            68: "microwave",
            69: "oven",
            70: "toaster",
            71: "sink",
            72: "refrigerator",
            73: "book",
            74: "clock",
            75: "vase",
            76: "scissors",
            77: "teddy bear",
            78: "hair drier",
            79: "toothbrush",
        }

        if self.architecture == "yolo":
            if self.model_dir_path:
                onnx_model_path = os.path.join(
                    self.model_dir_path, self.model + ".onnx"
                )
                self.session = rt.InferenceSession(onnx_model_path, providers=[
                        #'TensorrtExecutionProvider',
                        'CUDAExecutionProvider',
                        'CPUExecutionProvider'
                    ])
                inp = self.session.get_inputs()[0]
                self.input_name = inp.name
                self.input_dtype = np.float16 if "float16" in inp.type else np.float32
                self.input_size = inp.shape[2]  # [1, 3, H, W]

                # Decide detect vs. segment. Ultralytics exports stamp the task
                # in the ONNX metadata; an explicit override wins when given.
                meta = self.session.get_modelmeta().custom_metadata_map
                if self.task_override in ("detect", "segment"):
                    task = self.task_override
                else:
                    task = (meta.get("task") or "detect").lower()
                self.is_segment = task == "segment"

                outs = self.session.get_outputs()
                print(
                    f"[ObjectDetectorONNX] Loaded: {onnx_model_path} | task {task} | "
                    f"input {self.input_size}px {inp.type} | "
                    f"outputs {[o.shape for o in outs]}",
                    flush=True,
                )
                if self.is_segment and len(outs) < 2:
                    raise ValueError(
                        "Segmentation task selected but the ONNX model has a single "
                        "output; expected a second mask-prototype output."
                    )
            else:
                raise ValueError("No model path defined for ONNX model.")
        elif self.architecture == "detectron":
            raise ValueError(
                "Detectron return type was not adapted. Implement it if needed."
            )
        else:
            raise ValueError("Unrecognised architecture.")

    def preprocess(self, image):
        # Maintain aspect ratio and pad the image to the required input size
        input_size = self.input_size
        h, w, _ = image.shape
        scale = min(input_size / h, input_size / w)
        nh, nw = int(h * scale), int(w * scale)
        image_resized = cv2.resize(image, (nw, nh))

        top = (input_size - nh) // 2
        bottom = input_size - nh - top
        left = (input_size - nw) // 2
        right = input_size - nw - left

        image_padded = cv2.copyMakeBorder(
            image_resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

        image_padded = image_padded.astype(self.input_dtype) / 255.0
        image_padded = np.transpose(image_padded, (2, 0, 1))  # Change to (C, H, W)
        image_padded = np.expand_dims(image_padded, axis=0)  # Add batch dimension
        return image_padded, scale, top, left

    def postprocess(
        self,
        detection,
        original_width,
        original_height,
        scale,
        pad_top,
        pad_left,
        input_size=1280,
        conf_threshold=0.5,
        protos=None,
    ):
        """Decode raw model output into a detection DataFrame.

        Returns ``(df, masks)`` where ``masks`` is ``None`` for detection models
        and an ``[N, H, W]`` boolean array (one instance mask per row, at the
        original image resolution) for segmentation models.
        """
        try:
            # Ensure detection is a numpy array
            if isinstance(detection, list):
                detection = detection[0]

            detection = np.asarray(detection)
            if detection.ndim == 3:
                detection = detection[0]

            # YOLO26 end2end output: [300, 6] = [x1, y1, x2, y2, conf, class_id],
            # NMS done in model. Segmentation models append 32 mask coefficients:
            # [300, 6 + nm] paired with a [nm, mh, mw] prototype tensor (protos).
            if detection.shape[1] == 6 or (protos is not None and detection.shape[1] > 6):
                mask = detection[:, 4] > conf_threshold
                detection = detection[mask]
                x1, y1, x2, y2 = detection[:, 0], detection[:, 1], detection[:, 2], detection[:, 3]
                confidences = detection[:, 4]
                class_indices = detection[:, 5].astype(int)
                coeffs = detection[:, 6:] if protos is not None else None
                if self.classes is not None and len(self.classes) > 0:
                    cm = np.isin(class_indices, self.classes)
                    x1, y1, x2, y2 = x1[cm], y1[cm], x2[cm], y2[cm]
                    confidences, class_indices = confidences[cm], class_indices[cm]
                    if coeffs is not None:
                        coeffs = coeffs[cm]
                x1 = np.clip((x1 - pad_left) / scale, 0, original_width)
                y1 = np.clip((y1 - pad_top) / scale, 0, original_height)
                x2 = np.clip((x2 - pad_left) / scale, 0, original_width)
                y2 = np.clip((y2 - pad_top) / scale, 0, original_height)
                filtered_names = [self.class_dict.get(c, str(c)) for c in class_indices]
                df = pd.DataFrame({"xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2,
                                   "confidence": confidences, "class": class_indices, "name": filtered_names})
                masks = None
                if coeffs is not None and len(coeffs) > 0:
                    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
                    masks = self.build_masks(
                        protos, coeffs, boxes_xyxy, scale, pad_top, pad_left,
                        original_width, original_height,
                    )
                return df, masks

            if detection.shape[1] != 85:
                raise ValueError("Detection tensor shape is incorrect.")

            boxes = detection[:, :4]  # Bounding boxes (cx, cy, w, h)
            confidences = detection[:, 4]  # Confidence scores
            class_probs = detection[:, 5:]  # Class probabilities

            # Filter out low confidence detections
            indices = np.where(confidences > conf_threshold)

            boxes = boxes[indices]
            confidences = confidences[indices]
            class_probs = class_probs[indices]

            # Get class indices
            class_indices = np.argmax(class_probs, axis=1)

            # Filter by allowed classes if specified
            if self.classes is not None and len(self.classes) > 0:
                class_mask = np.isin(class_indices, self.classes)
                boxes = boxes[class_mask]
                confidences = confidences[class_mask]
                class_indices = class_indices[class_mask]

            # Convert center coordinates to corner coordinates
            cx = boxes[:, 0]
            cy = boxes[:, 1]
            w = boxes[:, 2]
            h = boxes[:, 3]

            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2

            # Scale boxes back to the original image size and correct for padding
            x1 = (x1 - pad_left) / scale
            y1 = (y1 - pad_top) / scale
            x2 = (x2 - pad_left) / scale
            y2 = (y2 - pad_top) / scale

            # Ensure the coordinates are within image dimensions
            x1 = np.clip(x1, 0, original_width)
            y1 = np.clip(y1, 0, original_height)
            x2 = np.clip(x2, 0, original_width)
            y2 = np.clip(y2, 0, original_height)

            # Map class indices to class names
            filtered_names = [self.class_dict[c] for c in class_indices]

            result_df = pd.DataFrame(
                {
                    "xmin": x1,
                    "ymin": y1,
                    "xmax": x2,
                    "ymax": y2,
                    "confidence": confidences,
                    "class": class_indices,
                    "name": filtered_names,
                }
            )

            return result_df, None

        except Exception as e:
            print(f"An error occurred: {e}")
            raise

    def build_masks(
        self,
        protos,
        coeffs,
        boxes_xyxy,
        scale,
        pad_top,
        pad_left,
        original_width,
        original_height,
    ):
        """Turn mask prototypes + per-detection coefficients into instance masks.

        Args:
            protos: prototype tensor, ``[1, nm, mh, mw]`` or ``[nm, mh, mw]``
            coeffs: ``[N, nm]`` mask coefficients (one row per detection)
            boxes_xyxy: ``[N, 4]`` detection boxes in original image coords
        Returns:
            ``[N, original_height, original_width]`` boolean array; each mask is
            cropped to its own bounding box so it can't bleed into neighbours.
        """
        protos = np.asarray(protos, dtype=np.float32)
        if protos.ndim == 4:
            protos = protos[0]
        nm, mh, mw = protos.shape

        # Linear combination of prototypes -> low-res mask logits in the
        # letterboxed input space, then squash to [0, 1].
        mask_logits = coeffs.astype(np.float32) @ protos.reshape(nm, -1)
        mask_prob = _sigmoid(mask_logits).reshape(-1, mh, mw)

        # Letterbox geometry used in preprocess(): the model input is the resized
        # image (nh x nw) centred with constant padding (pad_top, pad_left).
        input_size = self.input_size
        nh = int(original_height * scale)
        nw = int(original_width * scale)
        top = int(pad_top)
        left = int(pad_left)

        masks = np.zeros(
            (mask_prob.shape[0], original_height, original_width), dtype=bool
        )
        for i in range(mask_prob.shape[0]):
            # Upscale prototype mask to the full padded input, strip the padding,
            # then resize the valid region back to the original image size.
            m = cv2.resize(
                mask_prob[i], (input_size, input_size), interpolation=cv2.INTER_LINEAR
            )
            m = m[top : top + nh, left : left + nw]
            if m.size == 0:
                continue
            m = cv2.resize(
                m, (original_width, original_height), interpolation=cv2.INTER_LINEAR
            )
            binary = m > 0.5

            # Crop to the detection box (standard ultralytics behaviour).
            x1, y1, x2, y2 = boxes_xyxy[i]
            x1 = max(int(np.floor(x1)), 0)
            y1 = max(int(np.floor(y1)), 0)
            x2 = min(int(np.ceil(x2)), original_width)
            y2 = min(int(np.ceil(y2)), original_height)
            if x2 <= x1 or y2 <= y1:
                continue
            masks[i, y1:y2, x1:x2] = binary[y1:y2, x1:x2]

        return masks

    def detect(self, image):
        if self.architecture == "yolo":
            original_height, original_width = image.shape[:2]
            input_image, scale, pad_top, pad_left = self.preprocess(image)
            outputs = self.session.run(None, {self.input_name: input_image})

            # Segmentation models emit two tensors: a 3D detection tensor and a
            # 4D mask-prototype tensor. Pick them out by rank so we don't depend
            # on output ordering.
            protos = None
            detection_out = outputs
            if self.is_segment:
                detection_out = None
                for out in outputs:
                    arr = np.asarray(out)
                    if arr.ndim == 4 and protos is None:
                        protos = arr
                    elif arr.ndim == 3 and detection_out is None:
                        detection_out = arr
                if detection_out is None:
                    detection_out = outputs[0]

            detection, masks = self.postprocess(
                detection_out, original_width, original_height, scale, pad_top,
                pad_left, conf_threshold=self.confident, protos=protos,
            )

            if not self.multiple_instance:
                keep = self.single_instance_indices(detection)
                detection = detection.iloc[keep].reset_index(drop=True)
                if masks is not None:
                    masks = masks[keep]

            # Drawing the bounding boxes on the image
            for index, row in detection.iterrows():
                cv2.rectangle(
                    image,
                    (int(row["xmin"]), int(row["ymin"])),
                    (int(row["xmax"]), int(row["ymax"])),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    image,
                    f"{row['name']} {row['confidence']:.2f}",
                    (int(row["xmin"]), int(row["ymin"]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (36, 255, 12),
                    2,
                )

            # Tint each instance mask onto the debug image (green overlay).
            if masks is not None:
                for m in masks:
                    if m.any():
                        image[m] = (0.5 * image[m] + 0.5 * np.array([0, 200, 0])).astype(
                            image.dtype
                        )

            return detection, image, masks

    def single_instance_indices(self, detection):
        """Positional indices of the highest-confidence row per class.

        Detections arrive sorted by confidence (NMS output), so keeping the
        first occurrence of each class matches the previous drop behaviour while
        returning indices we can also apply to the parallel mask array.
        """
        seen = set()
        keep = []
        for i in range(len(detection)):
            cls = detection["class"].iloc[i]
            if cls in seen:
                continue
            seen.add(cls)
            keep.append(i)
        return keep
