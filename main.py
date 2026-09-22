from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import fitz  
import os
import uuid
import base64
import pymupdf


try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

app = FastAPI()


os.makedirs("temp", exist_ok=True)


app.mount("/static", StaticFiles(directory="static"), name="static")


class Modification(BaseModel):
    page_num: int
    bbox: list[float]
    new_text: str
    size: float
    color: str
    font_name: str | None = None
    offset_x: float = 0.0  


class SaveRequest(BaseModel):
    filename: str
    modifications: list[Modification]


class OCRRequest(BaseModel):
    filename: str


def int_to_hex_color(color_int: int) -> str:
    """Convierte el entero de color de PyMuPDF a formato hexadecimal CSS."""
    if color_int < 0:
        return "#000000"
    hex_str = f"{color_int:06x}"
    return f"#{hex_str}"


def hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    """Convierte Hex CSS a tupla RGB (0-1) para PyMuPDF."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        return (0.0, 0.0, 0.0)
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def match_pdf_font(font_string: str | None, is_italic: bool = False, is_bold: bool = False) -> str:
    """
    Analiza el nombre y los flags de la fuente original para mapearla
    a la familia Base-14 de PyMuPDF más parecida:
      - Sans-serif (Arial, Calibri, Helvetica, Verdana, Segoe, etc.) -> Helvetica
      - Monoespaciada (Courier, Consolas, etc.)                     -> Courier
      - Serif (Times, Georgia, Garamond, etc.)                      -> Times
    Antes esta función SIEMPRE devolvía una variante de Times sin
    importar la fuente original detectada; por eso un PDF en Arial
    terminaba editado en Times New Roman.
    """
    f_lower = (font_string or "").lower()

    bold = is_bold or any(k in f_lower for k in ["bold", "black", "heavy", "bld", "-b", "negrita", "semibold", "demi"])
    italic = is_italic or any(k in f_lower for k in ["italic", "oblique", "ital", "-i", "cursiva"])


    monospace_keywords = ["courier", "consolas", "mono", "menlo", "code", "typewriter"]
    sans_keywords = [
        "arial", "helvetica", "calibri", "verdana", "tahoma", "segoe",
        "roboto", "opensans", "open sans", "notosans", "noto sans",
        "lato", "montserrat", "trebuchet", "franklin", "univers",
        "myriad", "gothic", "sans"
    ]
    serif_keywords = [
        "times", "georgia", "cambria", "garamond", "book antiqua",
        "bookman", "minion", "palatino", "cardo", "constantia", "serif"
    ]

    if any(k in f_lower for k in monospace_keywords):
        family = "cour"
    elif any(k in f_lower for k in sans_keywords):
        family = "helv"
    elif any(k in f_lower for k in serif_keywords):
        family = "tiro"
    elif not f_lower:

        family = "helv"
    else:

        family = "helv"

    suffix_map = {
        "cour": {"regular": "cour", "bold": "cobo", "italic": "coit", "bolditalic": "cobi"},
        "helv": {"regular": "helv", "bold": "hebo", "italic": "heit", "bolditalic": "hebi"},
        "tiro": {"regular": "tiro", "bold": "tibo", "italic": "tiit", "bolditalic": "tibi"},
    }

    if bold and italic:
        style = "bolditalic"
    elif bold:
        style = "bold"
    elif italic:
        style = "italic"
    else:
        style = "regular"

    return suffix_map[family][style]


def merge_ocr_spans(spans: list[dict]) -> list[dict]:
    """
    Agrupa palabras fragmentadas por el OCR que pertenezcan a la misma línea horizontal 
    y estén cerca, convirtiéndolas en bloques de texto limpios y unificados.
    """
    if not spans:
        return []

    sorted_spans = sorted(spans, key=lambda s: (round(s["bbox"][1] / 8) * 8, s["bbox"][0]))
    
    merged = []
    current = None

    for s in sorted_spans:
        if not current:
            current = dict(s)
            continue

        vertical_overlap = abs(current["bbox"][1] - s["bbox"][1]) < 6
        horizontal_gap = s["bbox"][0] - current["bbox"][2]
        same_size = abs(current["size"] - s["size"]) < 2

        if vertical_overlap and 0 <= horizontal_gap < 25 and same_size:
            current["text"] += " " + s["text"]
            current["bbox"] = [
                current["bbox"][0],
                min(current["bbox"][1], s["bbox"][1]),
                s["bbox"][2],
                max(current["bbox"][3], s["bbox"][3])
            ]
        else:
            merged.append(current)
            current = dict(s)

    if current:
        merged.append(current)

    return merged


@app.get("/")
async def read_index():
    return FileResponse("static/index.html")


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    filepath = f"temp/{unique_filename}"
    
    with open(filepath, "wb") as f:
        f.write(await file.read())
    
    doc = fitz.open(filepath)
    
    if doc.is_encrypted:
        doc.authenticate("")
        if doc.is_encrypted:
            doc.close()
            return {"error": "Este PDF tiene contraseña y no se puede procesar."}
    
    pages_data = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_dict = page.get_text("dict")
        blocks = page_dict["blocks"]
        
        pix = page.get_pixmap(dpi=150)
        img_b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
        
        page_info = {
            "page_num": page_num,
            "width": page_dict["width"],
            "height": page_dict["height"],
            "image": f"data:image/png;base64,{img_b64}",
            "spans": []
        }
        
        for b in blocks:
            if b.get("type") == 0:
                for l in b.get("lines", []):
                    for s in l.get("spans", []):
                        if s.get("text", "").strip():
                            page_info["spans"].append({
                                "text": s["text"],
                                "bbox": s["bbox"],
                                "font": s.get("font", "Times-Roman"),
                                "size": s.get("size", 12),
                                "color": int_to_hex_color(s.get("color", 0)),
                                "flags": s.get("flags", 0)
                            })
        
        page_info["spans"] = merge_ocr_spans(page_info["spans"])
        pages_data.append(page_info)
    
    doc.close()
    return {"filename": unique_filename, "pages": pages_data}


@app.post("/ocr")
async def process_ocr(request: OCRRequest):
    filepath = f"temp/{request.filename}"
    if not os.path.exists(filepath):
        return {"error": "Archivo no encontrado"}

    doc = fitz.open(filepath)
    pages_data = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        try:
            tp = page.get_textpage_ocr(flags=0, language="spa", full=True)
            page_dict = tp.extractDICT()
        except Exception:
            page_dict = page.get_text("dict")

        pix = page.get_pixmap(dpi=150)
        img_b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")

        raw_spans = []
        for b in page_dict.get("blocks", []):
            if b.get("type") == 0:
                for l in b.get("lines", []):
                    for s in l.get("spans", []):
                        if s.get("text", "").strip():
                            raw_spans.append({
                                "text": s["text"],
                                "bbox": s["bbox"],
                                "font": s.get("font", "Times-Roman"),
                                "size": s.get("size", 12),
                                "color": int_to_hex_color(s.get("color", 0)),
                                "flags": s.get("flags", 0)
                            })

        unified_spans = merge_ocr_spans(raw_spans)

        page_info = {
            "page_num": page_num,
            "width": page_dict["width"],
            "height": page_dict["height"],
            "image": f"data:image/png;base64,{img_b64}",
            "spans": unified_spans
        }
        pages_data.append(page_info)

    doc.close()
    return {"filename": request.filename, "pages": pages_data}


@app.post("/save")
async def save_pdf(request: SaveRequest):
    try:
        filepath = f"temp/{request.filename}"
        if not os.path.exists(filepath):
            return {"error": "El archivo original no existe en el servidor."}

        doc = fitz.open(filepath)
        
        if doc.is_encrypted:
            doc.authenticate("")

        for mod in request.modifications:
            if not mod.bbox or len(mod.bbox) < 4:
                continue

            page = doc.load_page(mod.page_num)
            original_rect = fitz.Rect(mod.bbox)

            if original_rect.is_empty or original_rect.is_infinite:
                continue

            detected_font = mod.font_name
            is_italic_flag = False
            is_bold_flag = False

            page_dict = page.get_text("dict")
            for b in page_dict.get("blocks", []):
                if b.get("type") == 0:
                    for l in b.get("lines", []):
                        for s in l.get("spans", []):
                            s_rect = fitz.Rect(s["bbox"])
                            if original_rect.intersects(s_rect):
                                detected_font = s.get("font", detected_font)
                                flags = s.get("flags", 0)
                                if flags & 2:
                                    is_italic_flag = True
                                if flags & 16:
                                    is_bold_flag = True

            
            annot = page.add_redact_annot(original_rect, fill=None)
            annot.update()
            page.apply_redactions(images=0, graphics=0)

            
            if HAS_OPENCV:
                try:
                    pix = page.get_pixmap(dpi=150, clip=original_rect)
                    if pix.width > 2 and pix.height > 2:
                        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))

                        if pix.n == 4:
                            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                        elif pix.n == 3:
                            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        else:
                            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                        mask = cv2.adaptiveThreshold(
                            gray, 255, 
                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                            cv2.THRESH_BINARY_INV, 
                            11, 4
                        )
                        
                        kernel = np.ones((3, 3), np.uint8)
                        mask = cv2.dilate(mask, kernel, iterations=1)

                        if cv2.countNonZero(mask) > 0:
                            inpainted = cv2.inpaint(img_bgr, mask, inpaintRadius=4, flags=cv2.INPAINT_TELEA)
                            is_success, buffer = cv2.imencode(".png", inpainted)
                            if is_success:
                                page.insert_image(original_rect, stream=buffer.tobytes())
                except Exception as cv_err:
                    print(f"Aviso: Inpainting omitido: {cv_err}")

            
            target_font = match_pdf_font(detected_font, is_italic=is_italic_flag, is_bold=is_bold_flag)
            rgb_color = hex_to_rgb(mod.color)

           
            baseline_y = original_rect.y1 - (original_rect.height * 0.15)
            insert_point = fitz.Point(original_rect.x0 + mod.offset_x, baseline_y)

            rc = page.insert_text(
                insert_point,
                mod.new_text,
                fontsize=mod.size,
                color=rgb_color,
                fontname=target_font
            )

            if rc < 0:
                bounded_rect = fitz.Rect(
                    original_rect.x0 + mod.offset_x,
                    original_rect.y0 - 2,
                    original_rect.x1 + mod.offset_x + 5,
                    original_rect.y1 + 4
                )
                page.insert_textbox(
                    bounded_rect,
                    mod.new_text,
                    fontsize=mod.size,
                    color=rgb_color,
                    fontname=target_font,
                    align=fitz.TEXT_ALIGN_LEFT
                )

        out_filename = f"edited_{request.filename}"
        out_path = f"temp/{out_filename}"

        doc.save(out_path)
        doc.close()

        return {"download_url": f"/download/{out_filename}"}

    except Exception as e:
        print(f"Error crítico en /save: {str(e)}")
        return {"error": f"Error interno al guardar: {str(e)}"}


@app.get("/download/{filename:path}")
async def download(filename: str):
    filepath = f"temp/{filename}"
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type='application/pdf', filename=filename)
    return {"error": "Archivo no encontrado"}