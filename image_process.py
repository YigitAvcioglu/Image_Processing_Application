import cv2
import numpy as np

class ImageEngine:
    @staticmethod
    def process_stack(original_image, operation_stack):

        if original_image is None:
            return None

        try:
            img = original_image.copy()
            img = np.ascontiguousarray(img)

            for op in operation_stack:
                img = ImageEngine.apply_single_op(img, op)
                img = np.ascontiguousarray(img)

            return img
        except Exception as e:
            print(f"Processing Error: {e}")
            return original_image  # Hata olursa orijinali dön

    @staticmethod
    def apply_single_op(img, op):
        action = op["action"]
        p = op["params"]

        # Yardımcı: Griye çevir
        def to_gray(im):
            return cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) if len(im.shape) == 3 else im

        # FUNDAMENTALS
        if action == "grayscale":
            return to_gray(img) if len(img.shape) == 3 else img

        if action == "negative":
            return 255 - img

        if action == "log":
            img_f = img.astype(np.float32)
            m_val = np.max(img_f)
            if m_val == 0: return img
            c = 255 / np.log(1 + m_val)
            log_img = c * np.log(1 + img_f)
            return np.clip(log_img, 0, 255).astype(np.uint8)

        if action == "bit_plane":
            g = to_gray(img)
            r = ((g >> p["plane"]) & 1) * 255
            return cv2.cvtColor(r, cv2.COLOR_GRAY2RGB)

        if action == "quantize":
            div = 256 // p["levels"]
            return (img // div) * div + (div // 2)

        if action == "flip":
            return cv2.flip(img, p["mode"])

        if action == "rotate":
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), p["angle"], 1)
            return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)

        if action == "scale":
            return cv2.resize(img, None, fx=p["x"], fy=p["y"], interpolation=cv2.INTER_CUBIC)

        # --- INTENSITY & SPATIAL ---
        if action == "hist_eq":
            if len(img.shape) == 3:
                yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
                yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
                return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
            return cv2.equalizeHist(img)

        if action == "gamma":
            t = np.array([((i / 255.0) ** (1.0 / p["value"])) * 255 for i in np.arange(0, 256)]).astype("uint8")
            return cv2.LUT(img, t)

        if action == "contrast":
            return cv2.convertScaleAbs(img, alpha=p["value"], beta=0)

        if action == "blur":
            return cv2.blur(img, (p["k"], p["k"]))

        if action == "gaussian":
            return cv2.GaussianBlur(img, (p["k"], p["k"]), 0)

        if action == "median":
            return cv2.medianBlur(img, p["k"])

        if action == "unsharp":
            g = cv2.GaussianBlur(img, (p["k"], p["k"]), 10.0)
            return cv2.addWeighted(img, 1.5, g, -0.5, 0)

        if action == "laplacian":
            r = cv2.Laplacian(img, cv2.CV_64F, ksize=p["k"])
            r = cv2.convertScaleAbs(r)
            if len(r.shape) == 2: r = cv2.cvtColor(r, cv2.COLOR_GRAY2RGB)
            return r

        if action == "sobel":
            g = to_gray(img)
            gx = cv2.Sobel(g, cv2.CV_16S, 1, 0, ksize=p["k"])
            gy = cv2.Sobel(g, cv2.CV_16S, 0, 1, ksize=p["k"])
            r = cv2.addWeighted(cv2.convertScaleAbs(gx), 0.5, cv2.convertScaleAbs(gy), 0.5, 0)
            return cv2.cvtColor(r, cv2.COLOR_GRAY2RGB)

        # --- MORPHOLOGY ---
        if action == "morph":
            g = to_gray(img) if p["op"] in ["gradient", "tophat", "blackhat", "skeleton"] else img
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (p["k"], p["k"]))
            o = p["op"]

            if o == "erosion": return cv2.erode(img, k, iterations=1)
            if o == "dilation": return cv2.dilate(img, k, iterations=1)
            if o == "opening": return cv2.morphologyEx(img, cv2.MORPH_OPEN, k)
            if o == "closing": return cv2.morphologyEx(img, cv2.MORPH_CLOSE, k)
            if o == "gradient": return cv2.cvtColor(cv2.morphologyEx(g, cv2.MORPH_GRADIENT, k), cv2.COLOR_GRAY2RGB)
            if o == "tophat": return cv2.cvtColor(cv2.morphologyEx(g, cv2.MORPH_TOPHAT, k), cv2.COLOR_GRAY2RGB)
            if o == "blackhat": return cv2.cvtColor(cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, k), cv2.COLOR_GRAY2RGB)
            if o == "boundary": return cv2.subtract(img, cv2.erode(img, k))
            if o == "skeleton":
                _, b = cv2.threshold(g, 127, 255, cv2.THRESH_BINARY)
                s = np.zeros(b.shape, np.uint8)
                e = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
                while True:
                    opn = cv2.morphologyEx(b, cv2.MORPH_OPEN, e)
                    tmp = cv2.subtract(b, opn)
                    erd = cv2.erode(b, e)
                    s = cv2.bitwise_or(s, tmp)
                    b = erd.copy()
                    if cv2.countNonZero(b) == 0: break
                return cv2.cvtColor(s, cv2.COLOR_GRAY2RGB)

        # --- SEGMENTATION ---
        if action == "thresh":
            g = to_gray(img)
            m = p["method"]
            if m == "global":
                _, r = cv2.threshold(g, p["val"], 255, cv2.THRESH_BINARY)
            elif m == "otsu":
                _, r = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            elif m == "adaptive_mean":
                r = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
            elif m == "adaptive_gauss":
                r = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            return cv2.cvtColor(r, cv2.COLOR_GRAY2RGB)

        if action == "canny":
            g = to_gray(img)
            return cv2.cvtColor(cv2.Canny(g, p["t1"], p["t2"]), cv2.COLOR_GRAY2RGB)

        if action == "hough":
            g = to_gray(img)
            edges = cv2.Canny(g, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
            if lines is not None:
                if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            return img

        if action == "components":
            g = to_gray(img)
            _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            num, _, st, _ = cv2.connectedComponentsWithStats(b, 8)
            if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            for i in range(1, num):
                x, y, w, h, a = st[i]
                if a > 50:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            return img

        return img